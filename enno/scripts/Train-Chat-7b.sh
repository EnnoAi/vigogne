python vigogne/train/train_sft.py \
--model_name_or_path "/mnt/ssd/models/hf/vigogne2-7b-chat" \
--train_file "./enno/datasets/data-train-chat.jsonl" \
--eval_file "./enno/datasets/data-eval-chat.jsonl" \
--output_dir "./enno/outputs/vigogne2-enno-chat-7b-sft-lora-8bit" \
--run_name vigogne2-enno-chat-7b-sft-lora \
--overwrite_output_dir \
--mode chat \
--model_max_length 2048 \
--preprocessing_num_workers 4 \
--dataloader_num_workers 1 \
--block_size 512 \
--pack_into_block \
--load_in_8bit \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--target_modules q_proj v_proj k_proj o_proj gate_proj down_proj up_proj \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 8 \
--num_train_epochs 40 \
--learning_rate 3e-4 \
--warmup_ratio 0.05 \
--weight_decay 0.01 \
--gradient_checkpointing \
--logging_steps 1 \
--logging_first_step true \
--save_strategy steps \
--save_steps 5 \
--save_total_limit 3 \
--evaluation_strategy steps \
--eval_steps 5 \
--report_to wandb \
--do_train \
--do_eval \
--compute_dtype bfloat16 \
--load_best_model_at_end yes

# Merge
python ./scripts/merge_lora_weights.py \
./enno/outputs/vigogne2-enno-chat-7b-sft-lora-8bit \
./enno/models/vigogne2-enno-chat-7b-sft-lora-8bit


train_file="./enno/datasets/data-train-chat.jsonl"
mode=chat
model_max_length=2048
model_name_or_path="/mnt/ssd/models/hf/vigogne2-7b-chat"
output_dir="./enno/outputs/vigogne2-enno-chat-7b-sft-lora-8bit"

per_device_train_batch_size=8
gradient_accumulation_steps=4

python ./vigogne/train/train_sft.py \
    --model_name_or_path $model_name_or_path \
    --train_file $train_file \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --mode $mode \
    --model_max_length $model_max_length \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 1 \
    --pack_into_block \
    --block_size "2048" \
    --load_in_8bit \
    --lora_r "64" \
    --lora_alpha "16" \
    --lora_dropout "0.05" \
    --target_modules "q_proj" "v_proj" "k_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs "3" \
    --learning_rate "1e-4" \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --weight_decay "0" \
    --torch_compile \
    --fp16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --log_level "info" \
    --logging_steps 2 \
    --logging_first_step true \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit "3" \
    --report_to "tensorboard" "wandb" \
    --compute_dtype bfloat16 \
    --do_train