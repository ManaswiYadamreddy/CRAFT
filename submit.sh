#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -N craft-train-textcondv4
#$ -j y
#$ -o craft-train-textcondv4.log
#$ -l gpus=1
#$ -l gpu_memory=16G
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
python train_textcond.py \
    --lq_dir /projectnb/cs585/projects/craft/data/train/LQ_images_512x512 \
    --hq_dir /projectnb/cs585/projects/craft/data/train/images512x512 \
    --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \
    --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
    --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
    --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
    --output_dir checkpoints/textcond_v4 \
    --mixed_precision bf16 \
    --batch_size 4 \
    --max_train_steps 50000

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

