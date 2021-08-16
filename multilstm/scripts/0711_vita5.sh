CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 0711_resnet20_train_rnnprop_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --problem cifar10_resnet20_sparse --save_path resnet20_train_rnnprop_sparse --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 0711_resnet20_train_rnnprop_sparse_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_momentum --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 0711_resnet20_train_rnnprop_momentum_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python train_rnnprop.py --problem cifar10_resnet20_sparse --save_path resnet20_train_rnnprop_sparse_momentum --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 0711_resnet20_train_rnnprop_sparse_momentum_GPU3.out &