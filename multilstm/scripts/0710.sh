CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_cl --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0710_resnet20_train_rnnprop_cl_GPU0.out &


CUDA_VISIBLE_DEVICES=1 nohup python train_rnnprop.py --problem cifar10_resnet56 --save_path resnet56_train_rnnprop --if_cl=False --if_mt=False —num_epochs=10000 --num_steps=100 > 0710_resnet56_train_rnnprop_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --problem cifar10_resnet20 --save_path resnet20_train_rnnprop_cl_momentum --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 > 0710_resnet20_train_rnnprop_cl_momentum_GPU0.out &



CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_eval_1_step10000 --seed 1 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_eval_1_step10000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_eval_1_step10000 --seed 2 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_eval_2_step10000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_eval_2_step10000 --seed 2 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_eval_2_step10000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_momentum_eval_1_step10000 --seed 1 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl_momentum/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_momentum_eval_1_step10000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_momentum_eval_2_step10000 --seed 2 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl_momentum/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_momentum_eval_1_step10000_GPU0.out &




CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 2 \
    --grad_acc 1 \
    --valid_batch_size 1 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 110

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_eval_3_step10000 --seed 3 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_eval_3_step10000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_momentum_eval_3_step10000 --seed 3 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl_momentum/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_momentum_eval_3_step10000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_eval_4_step10000 --seed 4 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_eval_4_step10000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_momentum_eval_4_step10000 --seed 4 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl_momentum/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_momentum_eval_4_step10000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_eval_5_step10000 --seed 5 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_eval_5_step10000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_momentum_eval_5_step10000 --seed 5 --num_epochs=1 --num_steps=10000 --path resnet20_train_rnnprop_cl_momentum/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_momentum_eval_5_step10000_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_eval_5_step75000 --seed 5 --num_epochs=1 --num_steps=75000 --path resnet20_train_rnnprop_cl/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_eval_5_step75000_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --problem cifar10_resnet20 --output_path resnet20_train_rnnprop_cl_momentum_eval_5_step75000 --seed 5 --num_epochs=1 --num_steps=75000 --path resnet20_train_rnnprop_cl_momentum/rp.l2l-0 > 0710_resnet20_train_rnnprop_cl_momentum_eval_5_step75000_GPU0.out &