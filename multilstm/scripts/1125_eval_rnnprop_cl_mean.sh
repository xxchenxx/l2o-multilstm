for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop_mean.py --path=mnist_multilstm_rnnprop_cl_5_step100/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_rnnprop_cl_5_step100_mean_eval_${i} --seed ${i} 
done