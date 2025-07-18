torchrun --nproc_per_node=4 \
         --master_port=23455 \
         train_Sup.py \
         --model_name bpt_vit_b \
         --drop_path 0.0 \
         --dataset DOG120 \
         --tuning_type "prompt" \
         --num_prompts 100 \
         --channels 75 \
         --epochs 100 \
         --batch_size 32 \
         --weight_decay 1e-2 \
         --wd_head 3.0 \
         --lr 5e-3 \
         --min_lr 1e-8 \
         --warmup_epochs 10 \
         --model_ema \
         --save_dir "./run/sup-shallow-DOG120"