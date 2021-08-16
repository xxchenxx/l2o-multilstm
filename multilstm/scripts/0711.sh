CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_cl_updated_trace --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_cl_updated_trace_GPU0.out &

CUDA_VISIBLE_DEVICES=5 nohup python train_rnnprop.py --problem cifar10_resnet20_lt --save_path resnet20_train_rnnprop_cl_updated_lt --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_cl_updated_lt_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_cl_updated_momentum --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_cl_updated_momentum_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python train_rnnprop.py --problem cifar10_resnet20_lt --save_path resnet20_train_rnnprop_cl_updated_lt_momentum --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0713_resnet20_train_rnnprop_cl_updated_lt_momentum_GPU7.out &



CUDA_VISIBLE_DEVICES=7 python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_cl_updated_trace --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100
