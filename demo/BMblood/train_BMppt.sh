# change dataloaders flag:copy in line 697
python train_dual.py --workers 1 --device 0 --batch 1 \
         --data demo/BMblood/BMppt.yaml --img 1024 --cfg models/detect/yolov9-c.yaml \
         --weights '' --name BMppt-woCopy-bacth1-yolov9-c --hyp hyp.scratch-high.yaml \
         --min-items 0 --epochs 30 --close-mosaic 15
