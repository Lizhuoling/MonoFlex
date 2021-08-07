exp_id=debug

CUDA_VISIBLE_DEVICES=1 python plain_train_net.py \
--batch_size 8 \
--config runs/monoflex.yaml \
--output output/$exp_id \
--scratch_backbone \