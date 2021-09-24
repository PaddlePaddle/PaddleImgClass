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
from __future__ import absolute_import, division, print_function

import time
import paddle
from ppcls.engine.train.utils import update_loss, update_metric, log_info
from ppcls.utils import profiler


def train_epoch(trainer, epoch_id, print_batch_step):
    tic = time.time()

    train_dataloader = trainer.train_dataloader if trainer.use_dali else trainer.train_dataloader(
    )
    for iter_id, batch in enumerate(train_dataloader):
        if iter_id >= trainer.max_iter:
            break
        profiler.add_profiler_step(trainer.config["profiler_options"])
        if iter_id == 5:
            for key in trainer.time_info:
                trainer.time_info[key].reset()
        trainer.time_info["reader_cost"].update(time.time() - tic)
        if trainer.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        batch_size = batch[0].shape[0]
        batch[1] = batch[1].reshape([-1, 1]).astype("int64")

        trainer.global_step += 1
        # image input
        if trainer.amp:
            with paddle.amp.auto_cast(custom_black_list={
                    "flatten_contiguous_range", "greater_than"
            }):
                out = forward(trainer, batch)
                loss_dict = trainer.train_loss_func(out, batch[1])
        else:
            out = forward(trainer, batch)

        # calc loss
        if trainer.config["DataLoader"]["Train"]["dataset"].get(
                "batch_transform_ops", None):
            loss_dict = trainer.train_loss_func(out, batch[1:])
        else:
            loss_dict = trainer.train_loss_func(out, batch[1])

        # step opt and lr
        if trainer.amp:
            scaled = trainer.scaler.scale(loss_dict["loss"])
            scaled.backward()
            trainer.scaler.minimize(trainer.optimizer, scaled)
        else:
            loss_dict["loss"].backward()
            trainer.optimizer.step()
        trainer.optimizer.clear_grad()
        trainer.lr_sch.step()

        # below code just for logging
        # update metric_for_logger
        update_metric(trainer, out, batch, batch_size)
        # update_loss_for_logger
        update_loss(trainer, loss_dict, batch_size)
        trainer.time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            log_info(trainer, batch_size, epoch_id, iter_id)
        tic = time.time()


def forward(trainer, batch):
    if not trainer.is_rec:
        return trainer.model(batch[0])
    else:
        return trainer.model(batch[0], batch[1])
