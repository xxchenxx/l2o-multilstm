for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_every_k.py --path=mnist_multilstm_dm_5/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path 0824_mnist_multilstm_dm_5_every_100_eval_${i} --seed ${i} --k 100
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_every_k.py --path=mnist_multilstm_dm_5/cw.l2l-0 --problem=mnist --num_steps=100000 --output_path 0824_mnist_multilstm_dm_5_every_100_100000_eval_${i} --seed ${i} --k 100
done


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_every_k.py --path=mnist_multilstm_dm_5/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path 0824_mnist_multilstm_dm_5_every_1000_eval_${i} --seed ${i} --k 1000
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_every_k.py --path=mnist_multilstm_dm_5/cw.l2l-0 --problem=mnist --num_steps=100000 --output_path 0824_mnist_multilstm_dm_5_every_1000_100000_eval_${i} --seed ${i} --k 1000
done


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_every_k.py --path=mnist_multilstm_cl_dm_5/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path 0821_mnist_multilstm_cl_dm_5_every_100_eval_${i} --seed ${i} --k 100
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_every_k.py --path=mnist_multilstm_cl_dm_5/cw.l2l-0 --problem=mnist --num_steps=100000 --output_path 0821_mnist_multilstm_cl_dm_5_every_100_100000_eval_${i} --seed ${i} --k 100
done


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_every_k.py --path=mnist_multilstm_cl_dm_5/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path 0821_mnist_multilstm_cl_dm_5_every_1000_eval_${i} --seed ${i} --k 1000
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm_every_k.py --path=mnist_multilstm_cl_dm_5/cw.l2l-0 --problem=mnist --num_steps=100000 --output_path 0821_mnist_multilstm_cl_dm_5_every_1000_100000_eval_${i} --seed ${i} --k 1000
done
