







python train_net.py --config-file configs/flickr/flickr1.yaml   --output-dir FLIKR_MIM_MLM_Hash  MODE.TRAIN_NAME "FLIKR_MIM_MLM_Hash" MODE.MIM True MODE.MLM True  MODE.MASK_SEP True   SOLVER.IMS_PER_BATCH 64  SOLVER.WARMUP_EPOCHS 20 SOLVER.STEPS '(50, 80)' SOLVER.NUM_EPOCHS 100


python train_net.py --config-file configs/cub/cub.yaml  --resume-from-suspension --without-timestamp --output-dir CUB_MIM_MLM_Hash  MODE.TRAIN_NAME "cub_MIM_MLM_Hash"  MODE.MIM True MODE.MLM True  MODE.MASK_SEP True SOLVER.IMS_PER_BATCH 32  SOLVER.WARMUP_EPOCHS 20 SOLVER.STEPS '(50, 80)' SOLVER.NUM_EPOCHS 100

python train_net.py --config-file configs/cuhkpedes/cuhkpedes.yaml --resume-from-suspension --without-timestamp --output-dir CUHK_MIM_MLM_Hash  MODE.TRAIN_NAME "CUHK_MIM_MLM_Hash"  MODE.MIM True MODE.MLM True  MODE.MASK_SEP True SOLVER.IMS_PER_BATCH 64  SOLVER.WARMUP_EPOCHS 20 SOLVER.STEPS '(50, 80)' SOLVER.NUM_EPOCHS 100

