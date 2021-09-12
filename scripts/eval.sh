exp_id=MonoFlex_direct_uncern2

python3 ./kitti-object-eval-python/evaluate.py evaluate \
    --label_path /home/twilight/twilight/Data/kitti/MonoFlex_kitti/training/label_2 \
    --label_split_file /home/twilight/twilight/Data/kitti/MonoFlex_kitti/training/ImageSets/val.txt \
    --current_class 0,1,2 \
    --coco False \
    --result_path output/$exp_id/inference/kitti_train/data \