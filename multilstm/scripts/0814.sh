git checkout original
git pull
CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_original_cl --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > original_cl.out &

git checkout multilstm
git pull
CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm_cl --problem=mnist --if_cl=True --if_mt=False --num_epochs=10000 --num_steps=100 --seed 1 > multilstm_cl.out &


git checkout original
git pull
CUDA_VISIBLE_DEVICES=0 nohup python evaluate_dm.py --path=mnist_original/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_original_eval --seed 1 > original_evaluate.out &

git checkout multilstm
git pull
CUDA_VISIBLE_DEVICES=1 nohup python evaluate_dm.py --path=mnist_multilstm/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_eval --seed 1 > multilstm_evaluate.out &


git checkout original
git pull
CUDA_VISIBLE_DEVICES=0 nohup python evaluate_dm.py --path=mnist_original_cl/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_original_cl_eval --seed 1 > original_evaluate_cl.out &

git checkout multilstm
git pull
CUDA_VISIBLE_DEVICES=1 nohup python evaluate_dm.py --path=mnist_multilstm_cl/cw.l2l-0 --problem=mnist --num_steps=10000 --output_path mnist_multilstm_cl_eval --seed 1 > multilstm_evaluate.out &
