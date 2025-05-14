nohup python -u train_CVACT_test.py \
  --lr 0.0001 \
  --batch-size 32 \
  --dist-url 'tcp://localhost:8081' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --epochs 50 \
  --save_path ./result_cvact_test \
  --op sam \
  --wd 0.03 \
  --mining \
  --dataset cvact \
  --cos \
  --dim 6144 \
  --asam \
  --rho 2.5 \
    >> result_cvact_test.log 2>&1 &