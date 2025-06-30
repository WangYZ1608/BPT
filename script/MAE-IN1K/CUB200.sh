# Shallow
torchrun --nproc_per_node=4 \
         --master_port=23455 \
         train_MAE.py \
         --model_name MAE_bpt_vit_b \
         --finetune "./ckpt/MAE/mae_pretrain_vit_base.pth" \
         --drop_path 0.0 \
         --dataset CUB200 \
         --tuning_type "prompt" \
         --num_prompts 100 \
         --channels 75 \
         --epochs 100 \
         --batch_size 32 \
         --weight_decay 5e-3 \
         --wd_head 0.5 \
         --lr 5e-2 \
         --min_lr 1e-8 \
         --warmup_epochs 10 \
         --model_ema \
         --save_dir "./run/shallow-CUB200/"

# deep
torchrun --nproc_per_node=4 \
         --master_port=23455 \
         train_MAE.py \
         --model_name MAE_bpt_deep_vit_b \
         --prompt_deep \
         --finetune "./ckpt/MAE/mae_pretrain_vit_base.pth" \
         --drop_path 0.0 \
         --dataset CUB200 \
         --tuning_type "prompt" \
         --num_prompts 50 \
         --channels 50 \
         --epochs 100 \
         --batch_size 32 \
         --weight_decay 5e-3 \
         --wd_head 0.5 \
         --lr 5e-2 \
         --min_lr 1e-8 \
         --warmup_epochs 10 \
         --model_ema \
         --save_dir "./run/deep-CUB200/"