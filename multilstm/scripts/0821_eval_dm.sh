for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_dm_5/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path 0821_mnist_multilstm_dm_5_eval_${i} --seed ${i} 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_dm_5/cw.l2l-0 --problem=mnist --num_steps=100000 --output_path 0821_mnist_multilstm_dm_5_100000_eval_${i} --seed ${i} 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_mean.py --path=mnist_multilstm_dm_5/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path 0821_mnist_multilstm_dm_5_mean_eval_${i} --seed ${i} 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_mean.py --path=mnist_multilstm_dm_5/cw.l2l-0 --problem=mnist --num_steps=100000 --output_path 0821_mnist_multilstm_dm_5_100000_mean_eval_${i} --seed ${i} 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_cl_dm_5/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path 0821_mnist_multilstm_cl_dm_5_eval_${i} --seed ${i} 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_cl_dm_5/cw.l2l-0 --problem=mnist --num_steps=100000 --output_path 0821_mnist_multilstm_cl_dm_5_100000_eval_${i} --seed ${i} 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_mean.py --path=mnist_multilstm_cl_dm_5/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path 0821_mnist_multilstm_cl_dm_5_mean_eval_${i} --seed ${i} 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_mean.py --path=mnist_multilstm_cl_dm_5/cw.l2l-0 --problem=mnist --num_steps=100000 --output_path 0821_mnist_multilstm_cl_dm_5_100000_mean_eval_${i} --seed ${i} 
done

