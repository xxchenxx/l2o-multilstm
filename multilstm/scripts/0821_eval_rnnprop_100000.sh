for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_rnnprop.py --path=mnist_multilstm_rnnprop_5/rp.l2l-0 --problem=mnist --num_steps=100000 --output_path 0821_mnist_multilstm_rnnprop_5_100000_eval_${i} --seed ${i} 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_rnnprop_mean.py --path=mnist_multilstm_rnnprop_5/rp.l2l-0 --problem=mnist --num_steps=100000 --output_path 0821_mnist_multilstm_rnnprop_5_100000_mean_eval_${i} --seed ${i} 
done