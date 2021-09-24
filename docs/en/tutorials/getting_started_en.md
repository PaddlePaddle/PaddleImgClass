# Getting Started
---
Please refer to [Installation](install_en.md) to setup environment at first, and prepare flower102 dataset by following the instruction mentioned in the [Quick Start](quick_start_en.md).

## 1. Training and Evaluation on CPU or Single GPU

If training and evaluation are performed on CPU or single GPU, it is recommended to use the `tools/train.py` and `tools/eval.py`.
For training and evaluation in multi-GPU environment on Linux, please refer to [2. Training and evaluation on Linux+GPU](#2-training-and-evaluation-on-linuxgpu).

<a name="1.1"></a>
## 1.1 Model training

After preparing the configuration file, The training process can be started in the following way.

```
python tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Arch.pretrained=False \
    -o Global.device=gpu
```

Among them, `-c` is used to specify the path of the configuration file, `-o` is used to specify the parameters needed to be modified or added, `-o Arch.pretrained=False` means to not using pre-trained models.
`-o Global.device=gpu` means to use GPU for training. If you want to use the CPU for training, you need to set `Global.device` to `cpu`.


Of course, you can also directly modify the configuration file to update the configuration. For specific configuration parameters, please refer to [Configuration Document](config_description_en.md).

* The output log examples are as follows:
    * If mixup or cutmix is used in training, top-1 and top-k (default by 5) will not be printed in the log:

    ```
    ...
    epoch:0  , train step:20   , loss: 4.53660, lr: 0.003750, batch_cost: 1.23101 s, reader_cost: 0.74311 s, ips: 25.99489 images/sec, eta: 0:12:43
    ...
    END epoch:1   valid top1: 0.01569, top5: 0.06863, loss: 4.61747,  batch_cost: 0.26155 s, reader_cost: 0.16952 s, batch_cost_sum: 10.72348 s, ips: 76.46772 images/sec.
    ...
    ```

    * If mixup or cutmix is not used during training, in addition to the above information, top-1 and top-k (The default is 5) will also be printed in the log:

    ```
    ...
    epoch:0  , train step:30  , top1: 0.06250, top5: 0.09375, loss: 4.62766, lr: 0.003728, batch_cost: 0.64089 s, reader_cost: 0.18857 s, ips: 49.93080 images/sec, eta: 0:06:18
    ...
    END epoch:0   train top1: 0.01310, top5: 0.04738, loss: 4.65124,  batch_cost: 0.64089 s, reader_cost: 0.18857 s, batch_cost_sum: 13.45863 s, ips: 49.93080 images/sec.
    ...
    ```

During training, you can view loss changes in real time through `VisualDL`,  see [VisualDL](../extension/VisualDL_en.md) for details.

### 1.2 Model finetuning

After configuring the configuration file, you can finetune it by loading the pretrained weights, The command is as shown below.

```
python tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Arch.pretrained=True \
    -o Global.device=gpu
```

Among them, `-o Arch.pretrained` is used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file. You can also set it into `True` to use pretrained weights that trained in ImageNet1k.

We also provide a lot of pre-trained models trained on the ImageNet-1k dataset. For the model list and download address, please refer to the [model library overview](../models/models_intro_en.md).

### 1.3 Resume Training

If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```
python tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.checkpoints="./output/MobileNetV3_large_x1_0/epoch_5" \
    -o Global.device=gpu
```

The configuration file does not need to be modified. You only need to add the `Global.checkpoints` parameter during training, which represents the path of the checkpoints. The parameter weights, learning rate, optimizer and other information will be loaded using this parameter.

**Note**:

* The `-o Global.checkpoints` parameter does not need to include the suffix of the checkpoints. The above training command will generate the checkpoints as shown below during the training process. If you want to continue training from the epoch `5`, Just set the `Global.checkpoints` to `../output/MobileNetV3_large_x1_0/epoch_5`, PaddleClas will automatically fill in the `pdopt` and `pdparams` suffixes.

    ```shell
    output
    ├── MobileNetV3_large_x1_0
    │   ├── best_model.pdopt
    │   ├── best_model.pdparams
    │   ├── best_model.pdstates
    │   ├── epoch_1.pdopt
    │   ├── epoch_1.pdparams
    │   ├── epoch_1.pdstates
        .
        .
        .
    ```


### 1.4 Model evaluation

The model evaluation process can be started as follows.

```bash
python tools/eval.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

The above command will use `./configs/quick_start/MobileNetV3_large_x1_0.yaml` as the configuration file to evaluate the model `./output/MobileNetV3_large_x1_0/best_model`. You can also set the evaluation by changing the parameters in the configuration file, or you can update the configuration with the `-o` parameter, as shown above.

Some of the configurable evaluation parameters are described as follows:
* `Arch.name`: Model name
* `Global.pretrained_model`: The path of the model file to be evaluated

**Note:** If the model is a dygraph type, you only need to specify the prefix of the model file when loading the model, instead of specifying the suffix, such as [1.3 Resume Training](#13-resume-training).

<a name="2"></a>
### 2. Training and evaluation on Linux+GPU

If you want to run PaddleClas on Linux with GPU, it is highly recommended to use `paddle.distributed.launch` to start the model training script(`tools/train.py`) and evaluation script(`tools/eval.py`), which can start on multi-GPU environment more conveniently.

### 2.1 Model training

After preparing the configuration file, The training process can be started in the following way. `paddle.distributed.launch` specifies the GPU running card number by setting `gpus`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml
```

The format of output log information is the same as above, see [1.1 Model training](#11-model-training) for details.

### 2.2 Model finetuning

After configuring the configuration file, you can finetune it by loading the pretrained weights, The command is as shown below.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Arch.pretrained=True
```

Among them, `Arch.pretrained` is set to `True` or `False`. It also can be used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file.

There contains a lot of examples of model finetuning in [Quick Start](./quick_start_en.md). You can refer to this tutorial to finetune the model on a specific dataset.

<a name="model_resume"></a>
### 2.3 Resume Training

If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Global.checkpoints="./output/MobileNetV3_large_x1_0/epoch_5" \
        -o Global.device=gpu
```

The configuration file does not need to be modified. You only need to add the `Global.checkpoints` parameter during training, which represents the path of the checkpoints. The parameter weights, learning rate, optimizer and other information will be loaded using this parameter as described in [1.3 Resume training](#13-resume-training).

### 2.4 Model evaluation

The model evaluation process can be started as follows.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    tools/eval.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

About parameter description, see [1.4 Model evaluation](#14-model-evaluation) for details.

<a name="model_infer"></a>
## 3. Use the pre-trained model to predict
After the training is completed, you can predict by using the pre-trained model obtained by the training, as follows:

```python
python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Infer.infer_imgs=dataset/flowers102/jpg/image_00001.jpg \
    -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

Among them:
+ `Infer.infer_imgs`: The path of the image file or folder to be predicted;
+ `Global.pretrained_model`: Weight file path, such as `./output/MobileNetV3_large_x1_0/best_model`;

## 4. Use the inference model to predict

PaddlePaddle supports inference using prediction engines, which will be introduced next.

Firstly, you should export inference model using `tools/export_model.py`.

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.pretrained_model=output/MobileNetV3_large_x1_0/best_model
```

Among them,  `Global.pretrained_model` parameter is used to specify the model file path that does not need to include the file suffix name.

The above command will generate the model structure file (`inference.pdmodel`) and the model weight file (`inference.pdiparams`), and then the inference engine can be used for inference:

Go to the deploy directory:

```
cd deploy
```

Using inference engine to inference. Because the mapping file of ImageNet1k dataset is used by default, we should set `PostProcess.Topk.class_id_map_file` into `None`.

```bash
python3 python/predict_cls.py \
    -c configs/inference_cls.yaml \
    -o Global.infer_imgs=../dataset/flowers102/jpg/image_00001.jpg \
    -o Global.inference_model_dir=../inference/ \
    -o PostProcess.Topk.class_id_map_file=None
```
Among them:
+ `Global.infer_imgs`: The path of the image file to be predicted;
+ `Global.inference_model_dir`: Model structure file path, such as `../inference/inference.pdmodel`;
+ `Global.use_tensorrt`: Whether to use the TesorRT, default by `False`;
+ `Global.use_gpu`: Whether to use the GPU, default by `True`
+ `Global.enable_mkldnn`: Wheter to use `MKL-DNN`, default by `False`. It is valid when `Global.use_gpu` is `False`.
+ `Global.use_fp16`: Whether to enable FP16, default by `False`;

**Note**: If you want to use `Transformer series models`, such as `DeiT_***_384`, `ViT_***_384`, etc., please pay attention to the input size of model, and need to set `resize_short=384`, `resize=384`.

If you want to evaluate the speed of the model, it is recommended to enable TensorRT to accelerate for GPU, and MKL-DNN for CPU.
