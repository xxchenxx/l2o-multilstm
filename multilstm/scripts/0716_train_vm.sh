CUDA_VISIBLE_DEVICES=4 nohup python train_rnnprop.py --problem cifar10_resnet20_rp --save_path resnet20_train_rnnprop_rp --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 > 0716_resnet20_train_rnnprop_rp_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python train_rnnprop.py --problem cifar10_resnet20_rp --save_path resnet20_train_rnnprop_rp_cl --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0716_resnet20_train_rnnprop_rp_cl_GPU5.out &

CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20_rp --save_path resnet20_train_rnnprop_rp_cl_hessian --if_cl=True --if_mt=False --if_hessian True --num_epochs=10000 --num_steps=100 > 0716_resnet20_train_rnnprop_rp_cl_hessian_GPU0.out &



CUDA_VISIBLE_DEVICES=6 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_lt --path resnet20_train_rnnprop_rp_cl/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_rp_cl_eval_cifar10_resnet20_lt --seed 1 &

CUDA_VISIBLE_DEVICES=5 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_lt --path resnet20_train_rnnprop_rp_cl/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_rp_cl_eval_cifar10_resnet20_lt --seed 1 &

CUDA_VISIBLE_DEVICES=1 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_rp --path resnet20_train_rnnprop_rp_cl/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_rp_cl_eval_cifar10_resnet20_rp --seed 1 &

CUDA_VISIBLE_DEVICES=2 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_lt --path resnet20_train_rnnprop_rp_cl_hessian/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_rp_cl_hessian_eval_cifar10_resnet20_lt --seed 1 &

CUDA_VISIBLE_DEVICES=3 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_rp --path resnet20_train_rnnprop_rp_cl_hessian/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_rp_cl_hessian_eval_cifar10_resnet20_rp --seed 1 &

CUDA_VISIBLE_DEVICES=1 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_reinit --path resnet20_train_rnnprop_rp_cl/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_rp_cl_eval_cifar10_resnet20_reinit --seed 1 &

CUDA_VISIBLE_DEVICES=3 nohup python evaluate_rnnprop.py --problem cifar10_resnet20_reinit --path resnet20_train_rnnprop_rp_cl_hessian/rp.l2l-0 --num_epochs=1 --num_steps=75000 --output_path resnet20_train_rnnprop_rp_cl_hessian_eval_cifar10_resnet20_reinit --seed 1 &