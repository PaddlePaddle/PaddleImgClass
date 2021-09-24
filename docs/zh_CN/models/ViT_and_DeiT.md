# ViT与DeiT系列

## 概述

ViT（Vision Transformer）系列模型是Google在2020年提出的，该模型仅使用标准的Transformer结构，完全抛弃了卷积结构，将图像拆分为多个patch后再输入到Transformer中，展示了Transformer在CV领域的潜力。[论文地址](https://arxiv.org/abs/2010.11929)。

DeiT（Data-efficient Image Transformers）系列模型是由FaceBook在2020年底提出的，针对ViT模型需要大规模数据集训练的问题进行了改进，最终在ImageNet上取得了83.1%的Top1精度。并且使用卷积模型作为教师模型，针对该模型进行知识蒸馏，在ImageNet数据集上可以达到85.2%的Top1精度。[论文地址](https://arxiv.org/abs/2012.12877)。




## 精度、FLOPS和参数量

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ViT_small_patch16_224 | 0.7769 | 0.9342 | 0.7785 | 0.9342 |      |      |
| ViT_base_patch16_224  | 0.8195 | 0.9617 | 0.8178 | 0.9613 |      |      |
| ViT_base_patch16_384  | 0.8414 | 0.9717 | 0.8420 | 0.9722 |      |      |
| ViT_base_patch32_384  | 0.8176 | 0.9613 | 0.8166 | 0.9613 |      |      |
| ViT_large_patch16_224 | 0.8323 | 0.9650 | 0.8306 | 0.9644 |      |      |
| ViT_large_patch16_384 | 0.8513 | 0.9736 | 0.8517 | 0.9736 |      |      |
| ViT_large_patch32_384 | 0.8153 | 0.9608 | 0.815  | -      |      |      |


| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeiT_tiny_patch16_224            | 0.721 | 0.911 | 0.722 | 0.911 |      |      |
| DeiT_small_patch16_224           | 0.798 | 0.950 | 0.799 | 0.950 |      |      |
| DeiT_base_patch16_224            | 0.818 | 0.956 | 0.818 | 0.956 |      |      |
| DeiT_base_patch16_384            | 0.829 | 0.962 | 0.829 | 0.972 |      |      |
| DeiT_tiny_distilled_patch16_224  | 0.745 | 0.919 | 0.745 | 0.919 |      |      |
| DeiT_small_distilled_patch16_224 | 0.812 | 0.954 | 0.812 | 0.954 |      |      |
| DeiT_base_distilled_patch16_224  | 0.833 | 0.965 | 0.834 | 0.965 |      |      |
| DeiT_base_distilled_patch16_384  | 0.852 | 0.972 | 0.852 | 0.972 |      |      |

关于Params、FLOPs、Inference speed等信息，敬请期待。
