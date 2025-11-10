export JUDGE_OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxx"

model_path="/path/to/Qwen3-VL"
modality="video"
max_num_frames=32
max_new_tokens=4096

model_args="pretrained=$model_path,modality=$modality,max_num_frames=$max_num_frames,max_new_tokens=$max_new_tokens"


task_name=vsibench
output_path="./result"


num_processes=1
accelerate launch --num_processes $num_processes -m lmms_eval \
    --model qwen3_vl_moe \
    --model_args $model_args \
    --tasks $task_name \
    --log_samples \
    --process_with_media \
    --output_path $output_path