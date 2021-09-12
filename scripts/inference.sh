exp_id=MonoFlex_direct_uncern2

CUDA_VISIBLE_DEVICES=0 python -B plain_train_net.py \
    --config runs/monoflex_direct.yaml \
    --output output/$exp_id \
    --ckpt ./output/$exp_id/model_moderate_best_direct.pth  \
    --eval \
    #--vis \
    #--debug \
    #--eval_iou \