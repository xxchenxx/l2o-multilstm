CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --save_path=lora --problem=lora --if_cl=False --if_mt=False --num_epochs=100 --num_steps=100 --unroll_length 4 &


CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --save_path=lora_cl --problem=lora --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --unroll_length 4 &




CUDA_VISIBLE_DEVICES=0 nohup python train_rnnprop.py --save_path=lora_cl --problem=lora --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --unroll_length 4 &



CUDA_VISIBLE_DEVICES=0 nohup python evaluate_rnnprop.py --save_path=lora_cl --problem=lora --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --unroll_length 4 &

CUDA_VISIBLE_DEVICES=1 nohup python evaluate_rnnprop.py --problem lora --path lora_cl/rp.l2l-0 --num_epochs=1 --num_steps=10000 --output_path lora_cl_eval --seed 1 &

