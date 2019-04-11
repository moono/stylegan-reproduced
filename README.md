# stylegan-reproduced
* This repoitory is intended to understand official StyleGAN code from [https://github.com/NVlabs/stylegan](https://github.com/NVlabs/stylegan)
* Most of the code is just copy of original source code
* Try to remove tflib and some if/else statements

## current status
* checking if training works

## Dataset
* Download from [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)

## Requirements
* Tensorflow >= 1.12
* tensorflow estimator && [tf.contrib.distribute] for multi-GPU

## Steps to reproduce
* To check the network is properly copied
    1. export pretrained weight from official code
    2. set variable names to current implementation
    3. try to generate from official weights (inference_from_official_weights.py)
    4. check result
    
| official output | current implementation |
|:---------------:|:----------------------:|
|![hard to load?][official-output]|![hard to load?][current-output]|

## Training
* run train_v2.py


[tf.contrib.distribute]: https://www.tensorflow.org/api_docs/python/tf/contrib/distribute
[official-output]: ./assets/example.png
[current-output]: ./assets/from-official-weights.png