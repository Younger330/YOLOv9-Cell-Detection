# change dataloaders flag:copy in line 697
# 打开copy-paste需要加一个参数'--augu-cp'

save_dir=BMblood/noCP-yolov9-c-FL

python train_dual.py --workers 2 --device 0 --batch 4 \
         --data demo/BMblood/BM_IN.yaml --img 1024 --cfg models/detect/yolov9-c.yaml \
         --weights '' --name ${save_dir} --hyp data/hyps/hyp.scratch-high.yaml \
         --min-items 0 --epochs 100 --close-mosaic 15
