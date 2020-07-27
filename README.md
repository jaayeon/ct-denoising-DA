>>>lp-mayo CNL & edsr
(253)CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --model edsr --batch_size 32 --dataset lp-mayo --use_cuda --multi_gpu --body_part C N L

(253)CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model edsr --batch_size 64 --dataset lp-mayo --use_cuda --multi_gpu --n_epochs 200 --body_part C N L 

>>>piglet & edsr
(253)CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py –model edsr –batch_size 32 –dataset piglet –use_cuda –multi_gpu 

>>>lp-mayo L & edsr
(local) python main.py –model edsr –batch_size 12 –dataset lp-mayo –use_cuda –body_part L


>test

>>>python main.py --mode test --dataset piglet --use_cuda

