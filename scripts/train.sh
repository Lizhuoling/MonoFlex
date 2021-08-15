exp_id=debug

CUDA_VISIBLE_DEVICES=2 python -B plain_train_net.py \
--batch_size 8 \
--config runs/monoflex.yaml \
--output output/$exp_id \
--backbone dla34 \
--seed 22 \
#--scratch_backbone \