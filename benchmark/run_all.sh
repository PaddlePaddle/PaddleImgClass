# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37
# 执行目录：需说明
# cd **
# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
# pip install ...
# 2 拷贝该模型需要数据、预训练模型
# 3 批量运行（如不方便批量，1，2需放到单个模型中）
 
model_mode_list=(MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0 EfficientNetB0 ShuffleNetV2_x1_0 DenseNet121 HRNet_W48_C SwinTransformer_tiny_patch4_window7_224 alt_gvt_base)
fp_item_list=(fp32)
bs_list=(32 64 96 128)
for model_mode in ${model_mode_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_list[@]};do
	    echo "index is speed, 1gpus, begin, ${model_name}"
	    run_mode=sp
	    CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 10 ${model_mode}     #  (5min)
	    sleep 10
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 10 ${model_mode} 
            sleep 10
            done
      done
done
