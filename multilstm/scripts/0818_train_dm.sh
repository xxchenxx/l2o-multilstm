CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm_cl_dm_5 --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 &