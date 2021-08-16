CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm --problem=mnist --if_cl=False --if_mt=False —num_epochs=10000 --num_steps=100 &



CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm_cl --problem=mnist --if_cl=True --if_mt=False —num_epochs=10000 --num_steps=100 > cl.out &

CUDA_VISIBLE_DEVICES=1 python train_dm.py --save_path=mnist_multilstm_cl --problem=mnist --if_cl=True --if_mt=False —num_epochs=10000 --num_steps=100



CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e1000_s100 --problem=mnist --if_cl=False --if_mt=False —num_epochs=1000 --num_steps=100 &

CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm_e10000_s1000 --problem=mnist --if_cl=False --if_mt=False —num_epochs=10000 --num_steps=1000 &


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_eval_e10000_s100_multilstm_c_seed${i} --seed=${i}
done


CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e1000_s100 --problem=mnist --if_cl=False --if_mt=False --num_epochs=1000 --num_steps=100 &

CUDA_VISIBLE_DEVICES=1 nohup python train_dm.py --save_path=mnist_multilstm_e10000_s100 --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 &

CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e100_s100 --problem=mnist --if_cl=False --if_mt=False --num_epochs=100 --num_steps=100 &

CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e1000_s100 --problem=mnist --if_cl=False --if_mt=False --num_epochs=1000 --num_steps=100 &


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_e100_s100/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_eval_e100_s100_multilstm_c_seed${i} --seed=${i}
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_e1000_s100/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_eval_e1000_s100_multilstm_c_seed${i} --seed=${i}
done


CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e100_s100_test --problem=mnist --if_cl=False --if_mt=False --num_epochs=100 --num_steps=100 &


CUDA_VISIBLE_DEVICES=0 nohup python train_dm.py --save_path=mnist_multilstm_e10000_s100_test --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100 &


for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_e100_s100_test/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_eval_e100_s100_test_multilstm_c_seed${i} --seed=${i}
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_e10000_s100_test/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_eval_e10000_s100_test_multilstm_c_seed${i} --seed=${i}
done

for i in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0 python evaluate_dm.py --path=mnist_multilstm_e10000_s1000/cw.l2l-0 --num_steps=10000 --problem=mnist --output_path=mnist_eval_e10000_s1000_multilstm_c_seed${i} --seed=${i}
done