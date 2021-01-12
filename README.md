Domain Adaptation in Phantom & Real CT Denoising  
===============
reference code is available in https://github.com/SSinyu/WGAN_VGG/blob/master/networks.py
---------------
* Training commands
    * base : Denoising without perceptual loss. Denoising model can be [dncnn, unet, edsr] and default model is unet
        ```
        python main.py --way base --model [dncnn, unet, edsr] --source ge 
        ```
    * rev : Denoising without perceptual loss. Gradient reversal of target domain is included. Denoising model can be [dncnn, unet, edsr] and default model is unet
        ```
        python main.py --way rev --model [dncnn, unet, edsr] --source ge --target mayo --test_every 500
        ```
    * wgan : Denoising with perceptual loss. Denoising model is wganvgg.
        ```
        python main.py --way wgan --source ge --target mayo 
        ```
    * wganrev : Denoising with perceptual loss. Gradient reversal of target domain is included. Denoising model is wganvgg.
        ```
        python main.py --way wganrev --source ge --target mayo --test_every 500
        ```
    * out2src : Denoising with fake_target low dataset. You have to specify the fake_dir_name of source dataset (only base name of dir, not the full path).
        ```
        python main.py --way wganrev --source ge --target mayo --domain_sync out2src --fake_dir fake_dir
        ```
    * ref2trg : Denoising with fake_target low & high dataset. You have to specify the fake_dir name (only base name of dir, not the full path).
        ```
        python main.py --way wganrev --source ge --target mayo --domain_sync ref2trg --fake_dir fake_dir
        ```
---------------
* Test commands
    * python main.py --mode test --target mayo --thickness 3

---------------
* Requirements
    * OS: The package development version is training on Linux and tested on Windows operating systems with Anaconda.
    * Python : 3.7.1
    * Pytorch : 1.4.0



Datasets
===============
* Source data : Phantom dataset (vendor : GE)
* Target data : Mayo dataset (vendor : SIEMENS)