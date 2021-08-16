CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --path=trained_models/rnnprop/rp.l2l-0 --num_epochs=1 --num_steps=10000 --problem=cifar10_resnet56 --output_path=cifar10_resnet56_1 --seed=1 > 0414_resnet56_momentum_GPU0.out &

CUDA_VISIBLE_DEVICES=1 python evaluate_rnnprop.py --path=trained_models/rnnprop/rp.l2l-0 --num_epochs=1 --num_steps=10000 --problem=cifar10_resnet18 --output_path=cifar10_resnet18_1 --seed=1

CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop --if_cl=False --if_mt=False —num_epochs=10000 --num_steps=100 > 0414_resnet20_train_rnnprop_GPU0.out &