# change dataloaders flag:copy in line 697
# 打开copy-paste需要加一个参数'--augu-cp'

save_dir=BMblood/yolov9-t

python train_dual.py --workers 2 --device 0 --batch 4 \
         --data demo/BMblood/BM_IN.yaml --img 1024 --cfg models/detect/yolov9-t.yaml \
         --weights '' --name ${save_dir} --hyp data/hyps/hyp.scratch-high.yaml \
         --min-items 0 --epochs 60 --close-mosaic 15
