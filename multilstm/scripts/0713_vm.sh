CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_cl --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_cl_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --problem cifar10_resnet20_lt --save_path resnet20_train_rnnprop_cl_lt --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_cl_lt_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_cl_momentum --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_cl_momentum_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python train_rnnprop.py --problem cifar10_resnet20_lt --save_path resnet20_train_rnnprop_cl_lt_momentum --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_cl_lt_momentum_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python train_rnnprop.py --problem cifar10_resnet20_lt --save_path resnet20_train_rnnprop_lt --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_lt_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_momentum --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_momentum_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python train_rnnprop.py --problem cifar10_resnet20_lt --save_path resnet20_train_rnnprop_lt_momentum --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_lt_momentum_GPU7.out &

