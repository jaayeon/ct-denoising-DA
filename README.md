Command Ex 
(use_cuda : default True, model : default edsr, body_part : default C,N,L(only for lp-mayo) | required : --dataset)
====
lp-mayo
'''
>>(253)CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --batch_size 32 --dataset lp-mayo --multi_gpu --body_part C N L

>>(253)CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model dncnn --batch_size 64 --dataset lp-mayo --use_cuda --multi_gpu --n_epochs 200 --body_part L --test_every 200
'''

piglet & edsr
'''
>>(253)CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py –model edsr –batch_size 32 –dataset piglet  –multi_gpu 
'''

test
'''
>>python main.py --mode test --dataset piglet --use_cuda
'''