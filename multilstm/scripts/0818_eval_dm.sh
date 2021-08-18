for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_dm.py --path=mnist_multilstm_cl_dm_5/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_cl_dm_5_eval_${i} --seed ${i} 
done