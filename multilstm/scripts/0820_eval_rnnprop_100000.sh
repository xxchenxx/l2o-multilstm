for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_rnnprop.py --path=mnist_multilstm_rnnprop_5/rp.l2l-0 --problem=mnist --num_steps=100000 --output_path mnist_multilstm_rnnprop_5_100000_eval_${i} --seed ${i} 
done