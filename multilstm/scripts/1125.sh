CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --save_path=mnist_multilstm_rnnprop_5_step100 --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 1125_train_rnnprop.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --save_path=mnist_multilstm_rnnprop_cl_5_step100 --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 1125_train_rnnprop_cl.out &
