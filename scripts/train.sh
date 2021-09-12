exp_id=debug

CUDA_VISIBLE_DEVICES=0 python -B plain_train_net.py \
--batch_size 8 \
--config runs/monoflex_direct.yaml \
--output output/$exp_id \
--backbone dla34 \
--seed 22 \
#--scratch_backbone \
#--debug \