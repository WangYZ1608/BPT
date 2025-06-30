# Shallow
torchrun --nproc_per_node=4 \
         --master_port=23455 \
         train_MAE.py \
         --model_name MAE_bpt_vit_b \
         --finetune "./ckpt/MAE/mae_pretrain_vit_base.pth" \
         --drop_path 0.0 \
         --dataset Flowers102 \
         --tuning_type "prompt" \
         --num_prompts 100 \
         --channels 75 \
         --epochs 100 \
         --batch_size 32 \
         --weight_decay 5e-3 \
         --wd_head 1.0 \
         --lr 1e-1 \
         --min_lr 1e-8 \
         --warmup_epochs 10 \
         --model_ema \
         --save_dir "./run/shallow-Flowers102"