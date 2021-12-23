import os
import paddle
import cv2

from ppcls.arch import build_model
from ppcls.arch.gears.identity_head import IdentityHead
from ppcls.utils.config import parse_config, parse_args
from ppcls.utils.save_load import load_dygraph_pretrain
from ppcls.utils.logger import init_logger
from ppcls.data import transform, create_operators


def build_gallery_layer(configs, feature_extractor):
    transform_configs = configs["IndexProcess"]["transform_ops"]
    preprocess_ops = create_operators(transform_configs)

    embedding_size = configs["Arch"]["Head"]["embedding_size"]
    batch_size = configs["IndexProcess"]["batch_size"]
    image_shape = configs["Global"]["image_shape"]
    image_shape.insert(0, batch_size)
    input_tensor = paddle.zeros(image_shape)

    image_root = configs["IndexProcess"]["image_root"]
    data_file = configs["IndexProcess"]["data_file"]
    delimiter = configs["IndexProcess"]["delimiter"]
    gallery_images = []
    gallery_docs = []
    gallery_labels = []

    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for ori_line in lines:
            line = ori_line.strip().split(delimiter)
            text_num = len(line)
            assert text_num >= 2, f"line({ori_line}) must be splitted into at least 2 parts, but got {text_num}"
            image_file = os.path.join(image_root, line[0])

            gallery_images.append(image_file)
            gallery_docs.append(ori_line.strip())
            gallery_labels.append(line[1].strip())
    batch_index = 0
    gallery_feature = paddle.zeros((len(gallery_images), embedding_size))
    for i, image_path in enumerate(gallery_images):
        image = cv2.imread(image_path)
        for op in preprocess_ops:
            image = op(image)
        input_tensor[batch_index] = image
        batch_index += 1
        if batch_index == batch_size or i == len(gallery_images) - 1:
            batch_feature = feature_extractor(input_tensor)["features"]
            for j in range(batch_index):
                feature = batch_feature[j]
                norm_feature = paddle.nn.functional.normalize(feature, axis=0)
                gallery_feature[i - batch_index + j] = norm_feature
    gallery_layer = paddle.nn.Linear(embedding_size, len(gallery_images), bias_attr=False)
    gallery_layer.set_state_dict({"weight": gallery_feature.T})
    return gallery_layer


class FuseModel(paddle.nn.Layer):
    def __init__(self, configs):
        super().__init__()
        self.feature_extractor = build_model(configs)
        load_dygraph_pretrain(self.feature_extractor, configs["Global"]["pretrained_model"])
        self.feature_extractor.eval()
        self.feature_extractor.head = IdentityHead()
        self.gallery_layer = build_gallery_layer(configs, self.feature_extractor)

    def forward(self, x):
        x = self.feature_extractor(x)["features"]
        x = paddle.nn.functional.normalize(x)
        x = self.gallery_layer(x)
        return x


def export_fuse_model(configs):
    fuse_model = FuseModel(configs)
    fuse_model.eval()
    save_path = configs["Global"]["save_inference_dir"]
    fuse_model.quanter.save_quantized_model(
        fuse_model,
        save_path,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + configs["Global"]["image_shape"],
                dtype='float32')
        ])


def main():
    args = parse_args()
    configs = parse_config(args.config)
    init_logger(name='gallery2fc')
    export_fuse_model(configs)


if __name__ == '__main__':
    main()
