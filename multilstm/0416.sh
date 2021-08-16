CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --path=trained_models/rnnprop/rp.l2l-0 --num_epochs=1 --num_steps=75000 --problem=cifar10_resnet20_sparse --output_path=cifar10_resnet20_sparse_rnnprop_1_full --seed=1 > 0416_cifar10_resnet20_lth_1_full_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python evaluate_rnnprop.py --path=trained_models/rnnprop/rp.l2l-0 --num_epochs=1 --num_steps=75000 --problem=cifar10_resnet20 --output_path=cifar10_resnet20_dense_rnnprop_1_full --seed=1 > 0416_cifar10_resnet20_dense_1_full_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20_sparse --save_path resnet20_train_rnnprop_cl_sparse --if_cl=True --if_mt=False —num_epochs=10000 --num_steps=100 > 0417_resnet20_sparse_train_rnnprop_cl_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_cl_dense --if_cl=True --if_mt=False —num_epochs=10000 --num_steps=100 > 0417_resnet20_dense_train_rnnprop_cl_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20_sparse --save_path resnet20_train_rnnprop_cl_sparse  --if_mt=False —num_epochs=10000 --num_steps=100 > 0417_resnet20_sparse_train_rnnprop_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_cl_dense  --if_mt=False —num_epochs=10000 --num_steps=100 > 0417_resnet20_dense_train_rnnprop_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20_sparse --save_path resnet20_train_rnnprop_sparse_1000  --if_mt=False —num_epochs=1000 --num_steps=1000 > 0424_resnet20_sparse_train_rnnprop_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_dense_1000  --if_mt=False —num_epochs=1000 --num_steps=1000 > 0424_resnet20_dense_train_rnnprop_GPU1.out &