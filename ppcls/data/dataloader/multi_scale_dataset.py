#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import os

from paddle.io import Dataset
from paddle.vision import transforms
import cv2
import warnings

from ppcls.data import preprocess
from ppcls.data.preprocess import transform
from ppcls.data.preprocess.ops.operators import DecodeImage
from ppcls.utils import logger


def create_operators(params):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    ops = []
    for operator in params:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = getattr(preprocess, op_name)(**param)
        ops.append(op)

    return ops


class MultiScaleDataset(Dataset):
    def __init__(
            self,
            image_root,
            cls_label_path,
            transform_ops=None, ):
        self._img_root = image_root
        self._cls_path = cls_label_path
        self.transform_ops = transform_ops
        # if transform_ops:
        #     self._transform_ops = create_operators(transform_ops)

        self.images = []
        self.labels = []
        self._load_anno()

    def _load_anno(self, seed=None):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for l in lines:
                l = l.strip().split(" ")
                self.images.append(os.path.join(self._img_root, l[0]))
                self.labels.append(np.int64(l[1]))
                assert os.path.exists(self.images[-1])


    def __getitem__(self, properties):
        # properites is a tuple, contains (width, height, index)
        img_width = properties[0]
        img_height = properties[1]
        index = properties[2]
        has_crop = False
        if self.transform_ops:
            for i in range(len(self.transform_ops)):
                op = self.transform_ops[i]
                if 'RandCropImage' in op:
                    warnings.warn("Multi scale dataset will crop image according to the multi scale resolution")
                    self.transform_ops[i]['RandCropImage'] = {'size': img_width}
                    has_crop = True
        if has_crop == False:
            raise RuntimeError("Multi scale dateset requests RandCropImage")
        self._transform_ops = create_operators(self.transform_ops)

        try:
            with open(self.images[index], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops) 
            img = img.transpose((2, 0, 1))
            return (img, self.labels[index])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[index], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels))
