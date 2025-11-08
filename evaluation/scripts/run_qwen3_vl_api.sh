export JUDGE_OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxx"
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxx"

model_version="qwen3-vl-235b-a22b-instruct"

modality="video"
max_num_frames=32
max_new_tokens=4096
model_args="model_version=$model_version,max_num_frames=$max_num_frames,max_new_tokens=$max_new_tokens,modality=$modality"


task_name=vsibench
output_path=./result

python -m lmms_eval \
    --model qwen3_vl_api \
    --model_args $model_args \
    --tasks $task_name \
    --log_samples \
    --process_with_media \
    --output_path $output_path