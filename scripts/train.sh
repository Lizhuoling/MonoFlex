exp_id=official_seed26_2

CUDA_VISIBLE_DEVICES=3 python -B plain_train_net.py \
--batch_size 8 \
--config runs/monoflex.yaml \
--output output/$exp_id \
--seed 26 \
#--scratch_backbone \