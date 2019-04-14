# stylegan-reproduced
* This repoitory is intended to understand official StyleGAN code from [https://github.com/NVlabs/stylegan](https://github.com/NVlabs/stylegan)
* Most of the code is just copy of original source code
* In this Repository, I tried to remove tflib and some if/else statements that is annoying to see
* If you need fully adjustable/customizable code with all parameter settings, go with official one

## Current status
* checking if training works

## Dataset
* Download & setup from [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)

## Requirements
* Tensorflow >= 1.13 
* tensorflow estimator and [tf.contrib.distribute] for multi-GPU

## Steps to reproduce

### To check the generator network is properly copied
1. export pretrained weight from official code
2. set variable names to current implementation
3. try to generate from official weights ([inference_example_code])
4. check the result
    
| official output | current implementation |
|:---------------:|:----------------------:|
|![][official-output]|![][current-output]|

### Training - progressive training
* Trying to use native tensorflow training framework ([tf.estimator])
* It recreate tf.estimator object every time it needs to change resolution or or clear the optimizer states
* run [training code]

### Key implementations in official code
* Check [implementations]

[tf.contrib.distribute]: https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/MirroredStrategy
[inference_example_code]: ./inference_from_official_weights.py
[official-output]: ./assets/example.png
[current-output]: ./assets/from-official-weights.png
[tf.estimator]: https://www.tensorflow.org/guide/estimators
[training code]: train.py
[implementations]: ./implementations.md