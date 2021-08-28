exp_id=Official_seed22_B

CUDA_VISIBLE_DEVICES=3 python -B plain_train_net.py \
    --config runs/monoflex.yaml \
    --ckpt ./output/$exp_id/model_moderate_best_soft.pth  \
    --eval \
    #--eval_iou \