# 原始图片不进行任何操作
data_yaml=demo/BMblood/val-EX/bm2-ori.yaml
save_dir=BMblood/EX-bm2
# 风格迁移后的效果
# data_yaml=demo/BMblood/val-EX/bm2-style.yaml
# save_dir=BMblood/EX-bm2-CAGAN-yolov9-c

python val.py --data ${data_yaml} \
  --img 1024 --batch 4 --conf 0.001 --iou 0.3 \
  --device 0 --weights 'runs/train/BMblood/CP-yolov9-c/weights/best.pt' \
  --save-json --name ${save_dir}
