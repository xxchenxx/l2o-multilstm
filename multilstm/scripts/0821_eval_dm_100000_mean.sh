for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_dm_mean.py --path=mnist_multilstm_dm_5/cw.l2l-0 --problem=mnist --num_steps=100000 --output_path mnist_multilstm_dm_5_mean_100000_eval_${i} --seed ${i} 
done