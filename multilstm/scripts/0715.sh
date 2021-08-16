CUDA_VISIBLE_DEVICES=1 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_lt --path resnet20_train_rnnprop_cl_lt/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_cl_lt_eval_cifar10_resnet20_lt --seed 1 &

CUDA_VISIBLE_DEVICES=2 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_lt --path resnet20_train_rnnprop_cl_momentum/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_cl_momentum_eval_cifar10_resnet20_lt --seed 1 &

CUDA_VISIBLE_DEVICES=3 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_lt --path resnet20_train_rnnprop_cl_lt_momentum/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output-path resnet20_train_rnnprop_cl_lt_momentum_eval_cifar10_resnet20_lt --seed 1 & 

CUDA_VISIBLE_DEVICES=4 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_lt --path resnet20_train_rnnprop/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_eval_cifar10_resnet20_lt --seed 1 &

CUDA_VISIBLE_DEVICES=5 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_lt --path resnet20_train_rnnprop_lt/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_lt_eval_cifar10_resnet20_lt --seed 1 &


CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20_rp --save_path resnet20_train_rnnprop_cl_hessian_rp --if_cl=True --if_mt=False --if_hessian True --num_epochs=10000 --num_steps=100 > 0715_resnet20_train_rnnprop_cl_hessian_rp_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --problem cifar10_resnet20_rp --save_path resnet20_train_rnnprop_hessian_rp --if_cl=False --if_mt=False --if_hessian True --num_epochs=10000 --num_steps=100 > 0715_resnet20_train_rnnprop_hessian_rp_GPU1.out &
