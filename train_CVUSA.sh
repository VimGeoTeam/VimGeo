nohup python -u train_CVUSA.py \
  --lr 0.0001 \
  --batch-size 32 \
  --dist-url 'tcp://localhost:8082' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --epochs 200 \
  --save_path ./result_cvusa \
  --op sam \
  --wd 0.03 \
  --mining \
  --dataset cvusa \
  --cos \
  --dim 6144 \
  --asam \
  --rho 2.5 \
    >> result_cvusa.log 2>&1 &