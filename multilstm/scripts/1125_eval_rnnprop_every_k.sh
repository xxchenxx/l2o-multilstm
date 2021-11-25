for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_rnnprop_every_k.py --path=mnist_multilstm_rnnprop_5_step100/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path 1125_mnist_multilstm_rnnprop_5_step100_every_100_eval_${i} --seed ${i} --k 100
done