# Unsupervised Domain Adaptation for Low-dose Computed Tomography Denoising
paper can be found in [here](https://ieeexplore.ieee.org/document/9969607)

![gitimg](https://user-images.githubusercontent.com/33833333/223956344-700903ca-42e6-417e-a12e-18450e58f61c.jpg)

## Train
* base : Denoising without reversal loss. Denoising model can be [dncnn, unet, edsr].
    ``` 
    python main.py --way base --model [dncnn, unet, edsr] --source ge --vgg_weight 0.1 --l_weight 1
    ```
* rev : Denoising with reversal loss. Gradient reversal of target domain is included. Denoising model can be [dncnn, unet, edsr]. you can choose domain classifer input(dc_input) and reversal stage(style_stage).
    ```
    python main.py --way rev --model [dncnn, unet, edsr] --source ge --target mayo --test_every 500 --vgg_weight 0.01 --l_weight 1 --rev_weight 0.1 --dc_mode [mse, bce, wss] --dc_input [img, noise, feature, c_img, c_noise, c_feature] --style_stage [1,2,3,4,5,6] (--content_randomization)
    ```
* wgan : Denoising with wasserstein loss. 
    ```
    python main.py --way wgan --source ge --target mayo --vgg_weight 0.1 --l_weight 1
    ```
* wganrev : Denoising with wasserstein loss and reversal loss. Gradient reversal of target domain is included. Denoising model is wganvgg. you can choose domain classifer input(dc_input) and reversal stage(style_stage).
    ```
    python main.py --way wganrev --source ge --target mayo --test_every 500 --vgg_weight 0.1 --l_weight 1 --rev_weight 0.001 --dc_mode [mse, bce, wss] --dc_input [img, noise, feature, c_img, c_noise, c_feature] --style_stage [1,2,3,4,5,6] (--content_randomization)
    ```

* out2src : Denoising with fake_target low dataset. You have to specify the fake_dir_name of source dataset (only base name of dir, not the full path).
    ```
    python main.py --way wganrev --source ge --target mayo --domain_sync out2src --fake_dir fake_dir
    ```
* ref2trg : Denoising with fake_target low & high dataset. You have to specify the fake_dir name (only base name of dir, not the full path).
    ```
    python main.py --way wganrev --source ge --target mayo --domain_sync ref2trg --fake_dir fake_dir
    ```

## Test
```
python main.py --mode test --target mayo --thickness 3
```


## Requirements

* OS: The package development version is training on Linux and tested on Windows operating systems with Anaconda.
* Python : 3.7.1
* Pytorch : 1.4.0


## Datasets

* Source data : Phantom dataset (vendor : GE)
* Target data : Mayo dataset (vendor : SIEMENS)
