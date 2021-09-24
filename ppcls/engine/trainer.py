# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

import time
import platform
import datetime
import argparse
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from visualdl import LogWriter

from ppcls.utils.check import check_gpu
from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger
from ppcls.utils.logger import init_logger
from ppcls.utils.config import print_config
from ppcls.data import build_dataloader
from ppcls.arch import build_model
from ppcls.arch import apply_to_static
from ppcls.loss import build_loss
from ppcls.metric import build_metrics
from ppcls.optimizer import build_optimizer
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from ppcls.utils.save_load import init_model
from ppcls.utils import save_load

from ppcls.data.utils.get_image_list import get_image_list
from ppcls.data.postprocess import build_postprocess
from ppcls.data import create_operators


class Trainer(object):
    def __init__(self, config, mode="train"):
        self.mode = mode
        self.config = config
        self.output_dir = self.config['Global']['output_dir']

        log_file = os.path.join(self.output_dir, self.config["Arch"]["name"],
                                f"{mode}.log")
        init_logger(name='root', log_file=log_file)
        print_config(config)
        # set device
        assert self.config["Global"]["device"] in ["cpu", "gpu", "xpu"]
        # self.device = paddle.set_device(self.config["Global"]["device"])
        self.device = paddle.set_device('gpu:2')
        # set dist
        self.config["Global"][
            "distributed"] = paddle.distributed.get_world_size() != 1
        if self.config["Global"]["distributed"]:
            dist.init_parallel_env()

        if "Head" in self.config["Arch"]:
            self.is_rec = True
        else:
            self.is_rec = False

        self.model = build_model(self.config["Arch"])
        # set @to_static for benchmark, skip this by default.
        apply_to_static(self.config, self.model)

        if self.config["Global"]["pretrained_model"] is not None:
            if self.config["Global"]["pretrained_model"].startswith("http"):
                load_dygraph_pretrain_from_url(
                    self.model, self.config["Global"]["pretrained_model"])
            else:
                load_dygraph_pretrain(
                    self.model, self.config["Global"]["pretrained_model"])

        if self.config["Global"]["distributed"]:
            self.model = paddle.DataParallel(self.model)

        self.vdl_writer = None
        if self.config['Global']['use_visualdl'] and mode == "train":
            vdl_writer_path = os.path.join(self.output_dir, "vdl")
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)
        logger.info('train with paddle {} and device {}'.format(
            paddle.__version__, self.device))
        # init members
        self.train_dataloader = None
        self.eval_dataloader = None
        self.gallery_dataloader = None
        self.query_dataloader = None
        self.eval_mode = self.config["Global"].get("eval_mode",
                                                   "classification")
        self.amp = True if "AMP" in self.config else False
        if self.amp and self.config["AMP"] is not None:
            self.scale_loss = self.config["AMP"].get("scale_loss", 1.0)
            self.use_dynamic_loss_scaling = self.config["AMP"].get(
                "use_dynamic_loss_scaling", False)
        else:
            self.scale_loss = 1.0
            self.use_dynamic_loss_scaling = False
        if self.amp:
            AMP_RELATED_FLAGS_SETTING = {
                'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
                'FLAGS_max_inplace_grad_add': 8,
            }
            paddle.fluid.set_flags(AMP_RELATED_FLAGS_SETTING)
        self.train_loss_func = None
        self.eval_loss_func = None
        self.train_metric_func = None
        self.eval_metric_func = None
        self.use_dali = self.config['Global'].get("use_dali", False)

    def train(self):
        # build train loss and metric info
        if self.train_loss_func is None:
            loss_info = self.config["Loss"]["Train"]
            self.train_loss_func = build_loss(loss_info)
        if self.train_metric_func is None:
            metric_config = self.config.get("Metric")
            if metric_config is not None:
                metric_config = metric_config.get("Train")
                if metric_config is not None:
                    self.train_metric_func = build_metrics(metric_config)

        if self.train_dataloader is None:
            self.train_dataloader = build_dataloader(
                self.config["DataLoader"], "Train", self.device, self.use_dali)

        step_each_epoch = len(self.train_dataloader)

        optimizer, lr_sch = build_optimizer(self.config["Optimizer"],
                                            self.config["Global"]["epochs"],
                                            step_each_epoch,
                                            self.model.parameters())

        print_batch_step = self.config['Global']['print_batch_step']
        save_interval = self.config["Global"]["save_interval"]

        best_metric = {
            "metric": 0.0,
            "epoch": 0,
        }
        # key:
        # val: metrics list word
        output_info = dict()
        time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }
        # global iter counter
        global_step = 0

        if self.config["Global"]["checkpoints"] is not None:
            metric_info = init_model(self.config["Global"], self.model,
                                     optimizer)
            if metric_info is not None:
                best_metric.update(metric_info)

        # for amp training
        if self.amp:
            scaler = paddle.amp.GradScaler(
                init_loss_scaling=self.scale_loss,
                use_dynamic_loss_scaling=self.use_dynamic_loss_scaling)

        tic = time.time()
        max_iter = len(self.train_dataloader) - 1 if platform.system(
        ) == "Windows" else len(self.train_dataloader)
        for epoch_id in range(best_metric["epoch"] + 1,
                              self.config["Global"]["epochs"] + 1):
            acc = 0.0
            train_dataloader = self.train_dataloader if self.use_dali else self.train_dataloader(
            )
            for iter_id, batch in enumerate(train_dataloader):
                if iter_id >= max_iter:
                    break
                if iter_id == 5:
                    for key in time_info:
                        time_info[key].reset()
                time_info["reader_cost"].update(time.time() - tic)
                if self.use_dali:
                    batch = [
                        paddle.to_tensor(batch[0]['data']),
                        paddle.to_tensor(batch[0]['label'])
                    ]
                batch_size = batch[0].shape[0]
                batch[1] = batch[1].reshape([-1, 1]).astype("int64")

                global_step += 1
                # image input
                if self.amp:
                    with paddle.amp.auto_cast(custom_black_list={
                            "flatten_contiguous_range", "greater_than"
                    }):
                        
                        out = self.forward(batch)
                        out = out.astype('float32')
                        # with open('/workspace/PaddleClas/op_no_gray.txt', 'w+t') as f:
                        #     print(self.model.forward.program_cache.last()[-1][1].program, file=f)
                        # import pdb; pdb.set_trace()
                        # self.model.forward.program_cache.last()[-1][1].program
                        loss_dict = self.train_loss_func(out, batch[1])
                        
                else:
                    out = self.forward(batch)

                # calc loss
                if self.config["DataLoader"]["Train"]["dataset"].get(
                        "batch_transform_ops", None):
                    loss_dict = self.train_loss_func(out, batch[1:])
                else:
                    loss_dict = self.train_loss_func(out, batch[1])

                for key in loss_dict:
                    if not key in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')
                    output_info[key].update(loss_dict[key].numpy()[0],
                                            batch_size)
                # calc metric
                if self.train_metric_func is not None:
                    metric_dict = self.train_metric_func(out, batch[-1])
                    for key in metric_dict:
                        if not key in output_info:
                            output_info[key] = AverageMeter(key, '7.5f')
                        output_info[key].update(metric_dict[key].numpy()[0],
                                                batch_size)

                # step opt and lr
                if self.amp:
                    scaled = scaler.scale(loss_dict["loss"])
                    scaled.backward()
                    scaler.minimize(optimizer, scaled)
                else:
                    loss_dict["loss"].backward()
                    optimizer.step()
                optimizer.clear_grad()
                lr_sch.step()

                time_info["batch_cost"].update(time.time() - tic)

                if iter_id % print_batch_step == 0:
                    lr_msg = "lr: {:.5f}".format(lr_sch.get_lr())
                    metric_msg = ", ".join([
                        "{}: {:.5f}".format(key, output_info[key].avg)
                        for key in output_info
                    ])
                    time_msg = "s, ".join([
                        "{}: {:.5f}".format(key, time_info[key].avg)
                        for key in time_info
                    ])

                    ips_msg = "ips: {:.5f} images/sec".format(
                        batch_size / time_info["batch_cost"].avg)
                    eta_sec = ((self.config["Global"]["epochs"] - epoch_id + 1
                                ) * len(self.train_dataloader) - iter_id
                               ) * time_info["batch_cost"].avg
                    eta_msg = "eta: {:s}".format(
                        str(datetime.timedelta(seconds=int(eta_sec))))
                    logger.info(
                        "[Train][Epoch {}/{}][Iter: {}/{}]{}, {}, {}, {}, {}".
                        format(epoch_id, self.config["Global"][
                            "epochs"], iter_id,
                               len(self.train_dataloader), lr_msg, metric_msg,
                               time_msg, ips_msg, eta_msg))

                    logger.scaler(
                        name="lr",
                        value=lr_sch.get_lr(),
                        step=global_step,
                        writer=self.vdl_writer)
                    for key in output_info:
                        logger.scaler(
                            name="train_{}".format(key),
                            value=output_info[key].avg,
                            step=global_step,
                            writer=self.vdl_writer)
                tic = time.time()
            if self.use_dali:
                self.train_dataloader.reset()
            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].avg)
                for key in output_info
            ])
            logger.info("[Train][Epoch {}/{}][Avg]{}".format(
                epoch_id, self.config["Global"]["epochs"], metric_msg))
            output_info.clear()

            # eval model and save model if possible
            if self.config["Global"][
                    "eval_during_train"] and epoch_id % self.config["Global"][
                        "eval_interval"] == 0:
                acc = self.eval(epoch_id)
                if acc > best_metric["metric"]:
                    best_metric["metric"] = acc
                    best_metric["epoch"] = epoch_id
                    save_load.save_model(
                        self.model,
                        optimizer,
                        best_metric,
                        self.output_dir,
                        model_name=self.config["Arch"]["name"],
                        prefix="best_model")
                logger.info("[Eval][Epoch {}][best metric: {}]".format(
                    epoch_id, best_metric["metric"]))
                logger.scaler(
                    name="eval_acc",
                    value=acc,
                    step=epoch_id,
                    writer=self.vdl_writer)

                self.model.train()

            # save model
            if epoch_id % save_interval == 0:
                save_load.save_model(
                    self.model,
                    optimizer, {"metric": acc,
                                "epoch": epoch_id},
                    self.output_dir,
                    model_name=self.config["Arch"]["name"],
                    prefix="epoch_{}".format(epoch_id))
                # save the latest model
                save_load.save_model(
                    self.model,
                    optimizer, {"metric": acc,
                                "epoch": epoch_id},
                    self.output_dir,
                    model_name=self.config["Arch"]["name"],
                    prefix="latest")

        if self.vdl_writer is not None:
            self.vdl_writer.close()

    def build_avg_metrics(self, info_dict):
        return {key: AverageMeter(key, '7.5f') for key in info_dict}

    @paddle.no_grad()
    def eval(self, epoch_id=0):
        self.model.eval()
        if self.eval_loss_func is None:
            loss_config = self.config.get("Loss", None)
            if loss_config is not None:
                loss_config = loss_config.get("Eval")
                if loss_config is not None:
                    self.eval_loss_func = build_loss(loss_config)
        if self.eval_mode == "classification":
            if self.eval_dataloader is None:
                self.eval_dataloader = build_dataloader(
                    self.config["DataLoader"], "Eval", self.device,
                    self.use_dali)

            if self.eval_metric_func is None:
                metric_config = self.config.get("Metric")
                if metric_config is not None:
                    metric_config = metric_config.get("Eval")
                    if metric_config is not None:
                        self.eval_metric_func = build_metrics(metric_config)

            eval_result = self.eval_cls(epoch_id)

        elif self.eval_mode == "retrieval":
            if self.gallery_dataloader is None:
                self.gallery_dataloader = build_dataloader(
                    self.config["DataLoader"]["Eval"], "Gallery", self.device,
                    self.use_dali)

            if self.query_dataloader is None:
                self.query_dataloader = build_dataloader(
                    self.config["DataLoader"]["Eval"], "Query", self.device,
                    self.use_dali)
            # build metric info
            if self.eval_metric_func is None:
                metric_config = self.config.get("Metric", None)
                if metric_config is None:
                    metric_config = [{"name": "Recallk", "topk": (1, 5)}]
                else:
                    metric_config = metric_config["Eval"]
                self.eval_metric_func = build_metrics(metric_config)
            eval_result = self.eval_retrieval(epoch_id)
        else:
            logger.warning("Invalid eval mode: {}".format(self.eval_mode))
            eval_result = None
        self.model.train()
        return eval_result

    def forward(self, batch):
        if not self.is_rec:
            out = self.model(batch[0])
        else:
            out = self.model(batch[0], batch[1])
        return out

    @paddle.no_grad()
    def eval_cls(self, epoch_id=0):
        output_info = dict()
        time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }
        print_batch_step = self.config["Global"]["print_batch_step"]

        metric_key = None
        tic = time.time()
        eval_dataloader = self.eval_dataloader if self.use_dali else self.eval_dataloader(
        )
        max_iter = len(self.eval_dataloader) - 1 if platform.system(
        ) == "Windows" else len(self.eval_dataloader)
        for iter_id, batch in enumerate(eval_dataloader):
            if iter_id >= max_iter:
                break
            if iter_id == 5:
                for key in time_info:
                    time_info[key].reset()
            if self.use_dali:
                batch = [
                    paddle.to_tensor(batch[0]['data']),
                    paddle.to_tensor(batch[0]['label'])
                ]
            time_info["reader_cost"].update(time.time() - tic)
            batch_size = batch[0].shape[0]
            batch[0] = paddle.to_tensor(batch[0]).astype("float32")
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")
            # image input
            out = self.forward(batch)
            # calc loss
            if self.eval_loss_func is not None:
                loss_dict = self.eval_loss_func(out, batch[-1])
                for key in loss_dict:
                    if not key in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')
                    output_info[key].update(loss_dict[key].numpy()[0],
                                            batch_size)
            # calc metric
            if self.eval_metric_func is not None:
                metric_dict = self.eval_metric_func(out, batch[-1])
                if paddle.distributed.get_world_size() > 1:
                    for key in metric_dict:
                        paddle.distributed.all_reduce(
                            metric_dict[key],
                            op=paddle.distributed.ReduceOp.SUM)
                        metric_dict[key] = metric_dict[
                            key] / paddle.distributed.get_world_size()
                for key in metric_dict:
                    if metric_key is None:
                        metric_key = key
                    if not key in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')

                    output_info[key].update(metric_dict[key].numpy()[0],
                                            batch_size)

            time_info["batch_cost"].update(time.time() - tic)

            if iter_id % print_batch_step == 0:
                time_msg = "s, ".join([
                    "{}: {:.5f}".format(key, time_info[key].avg)
                    for key in time_info
                ])

                ips_msg = "ips: {:.5f} images/sec".format(
                    batch_size / time_info["batch_cost"].avg)

                metric_msg = ", ".join([
                    "{}: {:.5f}".format(key, output_info[key].val)
                    for key in output_info
                ])
                logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                    epoch_id, iter_id,
                    len(self.eval_dataloader), metric_msg, time_msg, ips_msg))

            tic = time.time()
        if self.use_dali:
            self.eval_dataloader.reset()
        metric_msg = ", ".join([
            "{}: {:.5f}".format(key, output_info[key].avg)
            for key in output_info
        ])
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        # do not try to save best model
        if self.eval_metric_func is None:
            return -1
        # return 1st metric in the dict
        return output_info[metric_key].avg

    def eval_retrieval(self, epoch_id=0):
        self.model.eval()
        # step1. build gallery
        gallery_feas, gallery_img_id, gallery_unique_id = self._cal_feature(
            name='gallery')
        query_feas, query_img_id, query_query_id = self._cal_feature(
            name='query')

        # step2. do evaluation
        sim_block_size = self.config["Global"].get("sim_block_size", 64)
        sections = [sim_block_size] * (len(query_feas) // sim_block_size)
        if len(query_feas) % sim_block_size:
            sections.append(len(query_feas) % sim_block_size)
        fea_blocks = paddle.split(query_feas, num_or_sections=sections)
        if query_query_id is not None:
            query_id_blocks = paddle.split(
                query_query_id, num_or_sections=sections)
        image_id_blocks = paddle.split(query_img_id, num_or_sections=sections)
        metric_key = None

        if self.eval_metric_func is None:
            metric_dict = {metric_key: 0.}
        else:
            metric_dict = dict()
            for block_idx, block_fea in enumerate(fea_blocks):
                similarity_matrix = paddle.matmul(
                    block_fea, gallery_feas, transpose_y=True)
                if query_query_id is not None:
                    query_id_block = query_id_blocks[block_idx]
                    query_id_mask = (query_id_block != gallery_unique_id.t())

                    image_id_block = image_id_blocks[block_idx]
                    image_id_mask = (image_id_block != gallery_img_id.t())

                    keep_mask = paddle.logical_or(query_id_mask, image_id_mask)
                    similarity_matrix = similarity_matrix * keep_mask.astype(
                        "float32")
                else:
                    keep_mask = None

                metric_tmp = self.eval_metric_func(similarity_matrix,
                                                   image_id_blocks[block_idx],
                                                   gallery_img_id, keep_mask)

                for key in metric_tmp:
                    if key not in metric_dict:
                        metric_dict[key] = metric_tmp[key] * block_fea.shape[
                            0] / len(query_feas)
                    else:
                        metric_dict[key] += metric_tmp[key] * block_fea.shape[
                            0] / len(query_feas)

        metric_info_list = []
        for key in metric_dict:
            if metric_key is None:
                metric_key = key
            metric_info_list.append("{}: {:.5f}".format(key, metric_dict[key]))
        metric_msg = ", ".join(metric_info_list)
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        return metric_dict[metric_key]

    def _cal_feature(self, name='gallery'):
        all_feas = None
        all_image_id = None
        all_unique_id = None
        if name == 'gallery':
            dataloader = self.gallery_dataloader
        elif name == 'query':
            dataloader = self.query_dataloader
        else:
            raise RuntimeError("Only support gallery or query dataset")

        has_unique_id = False
        max_iter = len(dataloader) - 1 if platform.system(
        ) == "Windows" else len(dataloader)
        dataloader_tmp = dataloader if self.use_dali else dataloader()
        for idx, batch in enumerate(
                dataloader_tmp):  # load is very time-consuming
            if idx >= max_iter:
                break
            if idx % self.config["Global"]["print_batch_step"] == 0:
                logger.info(
                    f"{name} feature calculation process: [{idx}/{len(dataloader)}]"
                )
            if self.use_dali:
                batch = [
                    paddle.to_tensor(batch[0]['data']),
                    paddle.to_tensor(batch[0]['label'])
                ]
            batch = [paddle.to_tensor(x) for x in batch]
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")
            if len(batch) == 3:
                has_unique_id = True
                batch[2] = batch[2].reshape([-1, 1]).astype("int64")
            out = self.model(batch[0], batch[1])
            batch_feas = out["features"]

            # do norm
            if self.config["Global"].get("feature_normalize", True):
                feas_norm = paddle.sqrt(
                    paddle.sum(paddle.square(batch_feas), axis=1,
                               keepdim=True))
                batch_feas = paddle.divide(batch_feas, feas_norm)

            if all_feas is None:
                all_feas = batch_feas
                if has_unique_id:
                    all_unique_id = batch[2]
                all_image_id = batch[1]
            else:
                all_feas = paddle.concat([all_feas, batch_feas])
                all_image_id = paddle.concat([all_image_id, batch[1]])
                if has_unique_id:
                    all_unique_id = paddle.concat([all_unique_id, batch[2]])
        if self.use_dali:
            dataloader_tmp.reset()
        if paddle.distributed.get_world_size() > 1:
            feat_list = []
            img_id_list = []
            unique_id_list = []
            paddle.distributed.all_gather(feat_list, all_feas)
            paddle.distributed.all_gather(img_id_list, all_image_id)
            all_feas = paddle.concat(feat_list, axis=0)
            all_image_id = paddle.concat(img_id_list, axis=0)
            if has_unique_id:
                paddle.distributed.all_gather(unique_id_list, all_unique_id)
                all_unique_id = paddle.concat(unique_id_list, axis=0)

        logger.info("Build {} done, all feat shape: {}, begin to eval..".
                    format(name, all_feas.shape))
        return all_feas, all_image_id, all_unique_id

    @paddle.no_grad()
    def infer(self, ):
        total_trainer = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        image_list = get_image_list(self.config["Infer"]["infer_imgs"])
        # data split
        image_list = image_list[local_rank::total_trainer]

        preprocess_func = create_operators(self.config["Infer"]["transforms"])
        postprocess_func = build_postprocess(self.config["Infer"][
            "PostProcess"])

        batch_size = self.config["Infer"]["batch_size"]

        self.model.eval()

        batch_data = []
        image_file_list = []
        for idx, image_file in enumerate(image_list):
            with open(image_file, 'rb') as f:
                x = f.read()
            for process in preprocess_func:
                x = process(x)
            batch_data.append(x)
            image_file_list.append(image_file)
            if len(batch_data) >= batch_size or idx == len(image_list) - 1:
                batch_tensor = paddle.to_tensor(batch_data)
                out = self.model(batch_tensor)
                if isinstance(out, list):
                    out = out[0]
                result = postprocess_func(out, image_file_list)
                print(result)
                batch_data.clear()
                image_file_list.clear()
