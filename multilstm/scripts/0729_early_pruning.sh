resnet20_train_rnnprop

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --path resnet20_train_rnnprop/rp.l2l-0 --num_epochs=1 --num_steps=10000 --output_path resnet20_train_rnnprop_eval_cifar10_resnet20_early --seed 1 &