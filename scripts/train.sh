exp_id=MonoFlex_vit_patch8

CUDA_VISIBLE_DEVICES=2 python -B plain_train_net.py \
--batch_size 8 \
--config runs/monoflex_vit.yaml \
--output output/$exp_id \
--backbone vit_small \
--seed 22 \
--lr 1e-4 \
#--debug \
#--scratch_backbone \