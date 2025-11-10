export JUDGE_OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxx"

model_path="/path/to/InternVL3.5"
modality="video"
num_segments=32
max_new_tokens=4096

model_args="pretrained=$model_path,modality=$modality,num_segments=$num_segments,max_new_tokens=$max_new_tokens"

task_name="vsibench"
output_path="./result"

num_processes=8
accelerate launch --num_processes $num_processes -m lmms_eval \
    --model internvl3 \
    --model_args $model_args \
    --tasks $task_name \
    --log_samples \
    --process_with_media \
    --output_path $output_path