CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --save_path=mnist_multilstm_rnnprop_5_step20 --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=20 > 1023_train_rnnprop.out &


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_rnnprop.py --path=mnist_multilstm_rnnprop_5_step20/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_rnnprop_5_step20_eval_${i} --seed ${i} 
done


CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --save_path=mnist_multilstm_rnnprop_5_step500 --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=500 > 1023_train_rnnprop.out &


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_rnnprop.py --path=mnist_multilstm_rnnprop_5_step500/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_rnnprop_5_step500_eval_${i} --seed ${i} 
done



CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --save_path=mnist_original_rnnprop_step20 --problem=mnist --if_cl=False --if_mt=False --num_epochs=1000 --num_steps=20 > 1025_train_rnnprop.out &



for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_rnnprop.py --path=mnist_original_rnnprop_step20/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_original_rnnprop_step20_eval_${i} --seed ${i} 
done