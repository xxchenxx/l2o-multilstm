CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --save_path=mnist_original_cl_rnnprop --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > original_rnnprop.out &

CUDA_VISIBLE_DEVICES=0 python train_rnnprop.py --save_path=mnist_original_cl_rnnprop --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 