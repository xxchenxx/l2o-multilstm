CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e1000_s100_2lstm --problem=mnist --if_cl=False --if_mt=False --num_epochs=1000 --num_steps=100 &

CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm_e10000_s100_2lstm --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 &

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_e1000_s100_2lstm/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_multilstm_e1000_s100_2lstm_c_seed${i} --seed=${i}
done

CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e1000_s100_1lstm --problem=mnist --if_cl=False --if_mt=False --num_epochs=1000 --num_steps=100 &

CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e10000_s100_1lstm --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 &



for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_e10000_s100_1lstm/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_multilstm_e10000_s100_1lstm_c_seed${i} --seed=${i}
done

CUDA_VISIBLE_DEVICES=1 python train_dm.py --save_path=mnist_multilstm_e1000_s100_1lstm --problem=mnist --if_cl=False --if_mt=False --num_epochs=1000 --num_steps=100



CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e1000_s100_original --problem=mnist --if_cl=False --if_mt=False --num_epochs=1000 --num_steps=100 > original.out &


CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e1000_s100_1lstm --problem=mnist --if_cl=False --if_mt=False --num_epochs=1000 --num_steps=100 > 1lstm.out &

CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm_e10000_s100_original --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 &


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_e10000_s100_original/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_multilstm_e10000_s100_original_c_seed${i} --seed=${i}
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_e10000_s100_1lstm/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_multilstm_e10000_s100_1lstm_c_seed${i} --seed=${i}
done
