CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop_sliding.py --problem cifar10_resnet20_rp --save_path resnet20_rp_train_rnnprop_cl_sliding --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --threshold 0.05 > 0720_resnet20_rp_train_rnnprop_cl_sliding_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop_sliding.py --problem cifar10_resnet20_lt --save_path resnet20_lt_train_rnnprop_cl_sliding --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --threshold 0.05 > 0720_resnet20_lt_train_rnnprop_cl_sliding_GPU1.out &

