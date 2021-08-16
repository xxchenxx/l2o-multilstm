git checkout multilstm
git pull
CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm_cl --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > multilstm_cl.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > multilstm.out &


CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_cl --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > multilstm_cl.out &


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_cl/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_cl_eval_${i} --seed ${i} 
done


CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > multilstm.out &

git checkout multilstm
git pull
CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --save_path=mnist_multilstm_cl_rnnprop --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > multilstm_rnnprop.out &


git checkout original
git pull
CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --save_path=mnist_original_cl_rnnprop --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > original_rnnprop.out &




for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=1 python evaluate_dm.py --path=mnist_multilstm/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_eval_${i} --seed ${i} 
done




CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --save_path=mnist_multilstm_cl_rnnprop --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > multilstm_rnnprop.out &


git checkout original
git pull
CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --save_path=mnist_original_cl_rnnprop --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > original_rnnprop.out &


git checkout multilstm
git pull

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --save_path=mnist_multilstm_cl_rnnprop --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > multilstm_rnnprop.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --save_path=mnist_multilstm_cl_rnnprop_4 --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > multilstm_rnnprop_4.out &

git checkout original 
git pull
for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop.py --path=mnist_original_cl_rnnprop/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_original_cl_rnnprop_eval_${i} --seed ${i} 
done

git checkout multilstm
git pull
for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop.py --path=mnist_multilstm_cl_rnnprop/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_cl_rnnprop_eval_${i} --seed ${i} 
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_rnnprop.py --path=mnist_multilstm_cl_rnnprop_4/rp.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_cl_rnnprop_4_eval_${i} --seed ${i} 
done