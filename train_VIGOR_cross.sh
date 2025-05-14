nohup python -u train_vigor_cross.py \
  --lr 0.0001 \
  --batch-size 32 \
  --dist-url 'tcp://localhost:8087' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --epochs 50 \
  --save_path ./result_vigor_cross_WBL \
  --op sam \
  --wd 0.03 \
  --mining \
  --dataset vigor \
  --cos \
  --cross \
  --dim 9600 \
  --asam \
  --rho 2.5 \
    >> result_vigor_cross_WBL.log 2>&1 &