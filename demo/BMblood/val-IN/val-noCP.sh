save_dir=BMblood/noCP-yolov9-c-FL

python val.py --data demo/BMblood/BM_IN.yaml \
    --img 1024 --batch 4 --conf 0.001 --iou 0.3 \
    --device 0 --weights 'runs/train/BMblood/noCP-yolov9-c-FL/weights/best.pt' \
    --save-json --name ${save_dir}

