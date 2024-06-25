
###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-06-02 06:15:14
### 
num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
yaml=${3:-"csqa"}


CUDA_VISIBLE_DEVICES=${gpus} llamafactory-cli eval config/${yaml}.yaml