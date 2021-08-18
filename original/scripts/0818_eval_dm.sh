for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_dm.py --path=mnist_original_dm/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_original_dm_eval_${i} --seed ${i} 
done