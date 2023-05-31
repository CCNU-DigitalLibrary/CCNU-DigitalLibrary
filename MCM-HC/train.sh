#!/usr/bin/env bash
set -x

PYTHON=${PYTHON:-"python"}

PY_ARGS=${@:1}

GPUS=${GPUS:-2}

while true # find unused tcp port`
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done


#
#CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file configs/cuhkpedes/debug.yaml --use-tensorboard \
#  --without-timestamp --resume-from-suspension \
#  --output-dir CUHK_MIM02_MLM02_CODEBOOK800_VVQ02_TVQ02_VQBETA015 \
#  MODE.TRAIN_NAME "CUHK_MIM02_MLM02_CODEBOOK800_VVQ02_TVQ02_VQBETA015" \
#  MODE.MIM True MODE.MLM True MODE.CODEBOOK True MODE.CODEBOOK_NUM 800 MODE.VQ_BETA 0.15 SOLVER.IMS_PER_BATCH 32

#CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file configs/cuhkpedes/debug.yaml --use-tensorboard \
#  --without-timestamp --resume-from-suspension \
#  --output-dir CUHK_MIM02_MLM02_CODEBOOK800_dvae_VVQ02_TVQ00_VQBETA015 \
#  MODE.TRAIN_NAME "CUHK_MIM02_MLM02_CODEBOOK800_dvae_VVQ02_TVQ00_VQBETA015" \
#  MODE.MIM True MODE.MLM True MODE.CODEBOOK None MODE.CODEBOOK_TYPE "dvae" MODE.CODEBOOK_NUM 800 SOLVER.IMS_PER_BATCH 32 \
#  MODE.VQ_BETA 0.15 MODE.VISUAL_VQ_W 0.2 MODE.TEXTUAL_VQ_W 0.2

#


# description

CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file configs/cuhkpedes/debug.yaml --use-tensorboard \
  --without-timestamp --resume-from-suspension \
  --output-dir CUHK_MIM02_MLM02_CODEBOOK800_dvae_VVQ02_TVQ00_VQBETA015 \
  MODE.TRAIN_NAME "CUHK_MIM02_MLM02_CODEBOOK800_dvae_VVQ02_TVQ00_VQBETA015" \
  MODE.MIM True MODE.MLM True MODE.CODEBOOK True MODE.CODEBOOK_TYPE "dvae" MODE.CODEBOOK_NUM 800 SOLVER.IMS_PER_BATCH 32 \
  MODE.VQ_BETA 0.15 MODE.VISUAL_VQ_W 0.2 MODE.TEXTUAL_VQ_W 0.0


# description
# use visual_vqloss and mlm mim loss(all weights is 0.2)
# codebook_type dalle dvae
# codebook_num=800
# warm_up 40 (80, 120) total_epoch=160
#CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file configs/cuhkpedes/debug.yaml --use-tensorboard \
#  --without-timestamp --resume-from-suspension \
#  --output-dir CUHK_MIM02_MLM02_CODEBOOK800_dvae_VVQ02_TVQ00_VQBETA1 \
#  MODE.TRAIN_NAME "CUHK_MIM02_MLM02_CODEBOOK800_dvae_VVQ02_TVQ00_VQBETA1" \
#  MODE.MIM True MODE.MLM True MODE.CODEBOOK True MODE.CODEBOOK_TYPE "dvae" MODE.CODEBOOK_NUM 800 SOLVER.IMS_PER_BATCH 32 \
#  MODE.VQ_BETA 1.0 MODE.VISUAL_VQ_W 0.2 MODE.TEXTUAL_VQ_W 0.0

