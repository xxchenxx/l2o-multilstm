for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_rnnprop.py --path=mnist_original_cl_rnnprop/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_original_cl_rnnprop_eval_10000_${i} --seed ${i} 
done