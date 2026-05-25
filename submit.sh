#!/bin/bash -l

#$ -l h_rt=8:00:00
#$ -N ksae_train
#$ -j y
#$ -o log/ksae_train.log
#$ -l gpus=1
#$ -l gpu_memory=24G
#$ -pe omp 8

# =============================================================================
# Load config (universal params + task list)
# =============================================================================
module load miniconda
conda activate craft-env

# python train_concat.py \
#     --lq_dir /projectnb/cs585/projects/craft/data/train/LQ_images_512x512 \
#     --hq_dir /projectnb/cs585/projects/craft/data/train/images512x512 \
#     --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \
#     --ckpt_path pretrained \
#     --output_dir checkpoints/concat_v1 \
#     --mixed_precision bf16 \
#     --batch_size 4 \
#     --max_train_steps 1
# python train_textcond.py \
#     --lq_dir /projectnb/cs585/projects/craft/data/train/LQ_images_512x512 \
#     --hq_dir /projectnb/cs585/projects/craft/data/train/images512x512 \
#     --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \
#     --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
#     --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
#     --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
#     --output_dir checkpoints/textcond_v5 \
#     --mixed_precision bf16 \
#     --batch_size 4 \
#     --max_train_steps 50000 \
#     --text_embed_mode eos \
#     --pixel_loss_weight 0 \
#     --grad_loss_weight 0

# python train_textcond_final.py \
#     --lq_dir /projectnb/cs585/projects/craft/data/train/LQ_images_512x512 \
#     --hq_dir /projectnb/cs585/projects/craft/data/train/images512x512 \
#     --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \
#     --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
#     --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
#     --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
#     --output_dir checkpoints/textcond_final \
#     --mixed_precision fp16 \
#     --batch_size 2 \
#     --max_train_steps 50000

# python ksae_collect_features.py \
#     --image_dir /projectnb/cs585/projects/craft/data/train/LQ_images_512x512 \
#     --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \
#     --output_dir features/ksae_dual \
#     --split train \
#     --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
#     --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
#     --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
#     --conditioner_path checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
#     --film_neg_weight 0.1 \
#     --batch_size 8 \
#     --gpu_ids 0 \
#     --mixed_precision fp16

# python ksae_collect_features.py \
#     --image_dir /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2 \
#     --prompts_json /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
#     --output_dir features/ksae_dual \
#     --split test \
#     --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
#     --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
#     --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
#     --conditioner_path checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
#     --film_neg_weight 0.1 \
#     --batch_size 8 \
#     --gpu_ids 0 \
#     --mixed_precision fp16

# python ksae_train.py \
#     --features_path features/ksae_dual/features_up1a1_train.npy \
#     --output_dir checkpoints/ksae_up1a1 \
#     --k 32 --expansion_factor 64 --lr 4e-4 \
#     --warmup_steps 500 --max_steps 2000000 \
#     --batch_size 2048 --save_every 200000 --gpu_ids 0
# Kill the current run, then relaunch with expansion_factor=8 instead of 64
# python ksae_train.py \
#     --features_path features/ksae_dual/features_up2a1_train.npy \
#     --output_dir checkpoints/ksae_up2a1 \
#     --k 16 \
#     --expansion_factor 2 \
#     --lr 4e-4 \
#     --warmup_steps 500 \
#     --max_steps 2000000 \
#     --batch_size 2048 \
#     --save_every 200000 \
#     --gpu_ids 0

# python ksae_train.py \
#     --features_path features/ksae_dual/features_up2a1_train.npy \
#     --output_dir checkpoints/ksae_up2a1 \
#     --k 32 --expansion_factor 8 --lr 4e-4 \
#     --warmup_steps 500 --max_steps 2000000 \
#     --batch_size 2048 --save_every 200000 --gpu_ids 0

python ksae_analyze.py \
    --ksae_path      checkpoints/ksae_up2a1/ksae_latest.pt \
    --features_path  features/ksae_dual/features_up2a1_test.npy \
    --attrs_path     features/ksae_dual/attrs_test.npy \
    --filenames_path features/ksae_dual/filenames_test.json \
    --image_dir      /projectnb/cs585/projects/craft/data/test/CelebA/self_celeba_512_v2 \
    --output_dir     results/ksae_analysis_up2a1 \
    --top_n 10 --top_neurons_to_visualize 10 --gpu_ids 0

python ksae_analyze.py \
    --ksae_path      checkpoints/ksae_up2a1/ksae_latest.pt \
    --features_path  features/ksae_dual/features_up2a1_train.npy \
    --attrs_path     features/ksae_dual/attrs_train.npy \
    --filenames_path features/ksae_dual/filenames_train.json \
    --image_dir      /projectnb/cs585/projects/craft/data/train/LQ_images_512x512 \
    --output_dir     results/ksae_analysis_up2a1_train \
    --top_n 10 \
    --top_neurons_to_visualize 10 \
    --gpu_ids 0
    # python train_concat.py \
    # --lq_dir /projectnb/cs585/projects/craft/data/train/LQ_images_512x512 \
    # --hq_dir /projectnb/cs585/projects/craft/data/train/images512x512 \
    # --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \
    # --pretrained_model_name_or_path pretrained/sd21 \
    # --img_encoder_weight pretrained/associate_2.ckpt \
    # --ckpt_path pretrained \
    # --output_dir checkpoints/concat_v2 \
    # --mixed_precision bf16 \
    # --batch_size 4 \
    # --max_train_steps 50000

