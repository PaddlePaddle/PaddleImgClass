# benchmark使用说明

此目录所有shell脚本是为了测试PaddleClas中不同模型的速度指标，如单卡训练速度指标、多卡训练速度指标等。

## 相关脚本说明

一共有3个脚本：

- `prepare_data.sh`: 下载相应的测试数据，并配置好数据路径
- `run_benchmark.sh`: 执行单独一个训练测试的脚本，具体调用方式，可查看脚本注释
- `run_all.sh`: 执行所有训练测试的入口脚本

## 使用说明

**注意**：为了跟PaddleClas中其他的模块的执行目录保持一致，此模块的执行目录为`PaddleClas`的根目录。

### 1.准备数据

```shell
bash benchmark/prepare_data.sh
```

### 2.执行所有模型的测试

```shell
bash benchmark/run_all.sh
```
