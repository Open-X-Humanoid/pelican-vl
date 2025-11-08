export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG

export JUDGE_OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxx"

model_path="/path/to/Qwen2.5-VL"
modality="video"
max_num_frames=32
max_new_tokens=4096


gpu_memory_utilization=0.9
tensor_parallel_size=8

vllm_args="use_vllm=True,tensor_parallel_size=$tensor_parallel_size,gpu_memory_utilization=$gpu_memory_utilization"
model_args="$vllm_args,pretrained=$model_path,modality=$modality,max_num_frames=$max_num_frames,max_new_tokens=$max_new_tokens"


task_name="vsibench"
output_path="./result"


python3 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args $model_args \
    --tasks $task_name \
    --log_samples \
    --process_with_media \
    --output_path $output_path