nohup python -u train_vigor.py \
  --lr 0.0001 \
  --batch-size 32 \
  --dist-url 'tcp://localhost:8089' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --epochs 50 \
  --save_path ./result_vigor_wbl \
  --op sam \
  --wd 0.03 \
  --mining \
  --dataset vigor \
  --cos \
  --dim 9600 \
  --asam \
  --rho 2.5 \
    >> result_vigor_wbl.log 2>&1 &
