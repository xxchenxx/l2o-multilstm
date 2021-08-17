for i in $(seq 1 1); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop.py --path=mnist_multilstm_cl_rnnprop_5/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_cl_rnnprop_5_eval_${i} --seed ${i} 
done