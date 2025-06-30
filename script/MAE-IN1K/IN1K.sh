# ------------------------------------------------------------- #
# scale data, batch1024-lr0.1-wd1e-3, cosine decay
# data size   100%    30%     10%     3%     1%
# wd_head     1e-2    1e-1    1e-1    5e-1   5e-1
# ------------------------------------------------------------- #
torchrun --nproc_per_node=8 \
         --master_port=23455 \
         train_MAE.py \
         --model_name MAE_bpt_vit_b \
         --finetune "./ckpt/MAE/mae_pretrain_vit_base.pth" \
         --drop_path 0.0 \
         --dataset ImageNet \
         --tuning_type "prompt" \
         --num_prompts 100 \
         --channels 25 \
         --epochs 100 \
         --batch_size 1024 \
         --workers 64 \
         --weight_decay 1e-3 \
         --wd_head 1e-2 \
         --lr 0.1 \
         --min_lr 1e-8 \
         --warmup_epochs 10 \
         --model_ema \
         --save_dir "./run/shallow-IN1K/"
