python detect.py --weights 'runs/train/BMblood/CP-yolov9-c/weights/best.pt' \
  --source=../data/BMBlood/250213_val/*/*.jpg \
  --img 1024 --conf-thres 0.01 --iou-thres 0.3 \
  --device 1  --save-txt
