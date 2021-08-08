CUDA_VISIBLE_DEVICES=3 python plain_train_net.py \
    --config runs/monoflex.yaml \
    --ckpt ./output/backbone_scratch/model_moderate_best_soft.pth  \
    --eval \