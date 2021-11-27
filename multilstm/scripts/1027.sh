
for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop_every_k.py --path=mnist_multilstm_rnnprop_5_step500/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_rnnprop_5_step500_eval_every_k${i} --seed ${i} --k 100 
done


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop_every_k.py --path=mnist_multilstm_rnnprop_5_step500/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_rnnprop_5_step500_eval_every_k${i} --seed ${i} --k 100 
done


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop_mean.py --path=mnist_multilstm_rnnprop_5_step500/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_rnnprop_5_step500_eval_mean_${i} --seed ${i} --k 100 
done



CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --save_path=mnist_multilstm_rnnprop_5_step100 --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 1023_train_rnnprop.out &


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop_mean.py --path=mnist_multilstm_rnnprop_5_step100/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_rnnprop_5_step100_eval_mean_${i} --seed ${i} --k 100 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop_every_k.py --path=mnist_multilstm_rnnprop_5_step100/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_rnnprop_5_step100_eval_every_k_${i} --seed ${i} --k 100 
done