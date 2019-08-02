# stylegan-reproduced
* This repoitory is intended to understand official StyleGAN code from [https://github.com/NVlabs/stylegan](https://github.com/NVlabs/stylegan)
* Most of the code is __just copy__ of original source code
* In this Repository, I tried to remove tflib and some if/else statements from official code

## Requirements
* Tensorflow >= 1.13 
* tensorflow estimator and [tf.contrib.distribute] for multi-GPU
* Training time: approx. 5 days 8 hours with V100 * 4
  * note: training step on resolution 1024x1024 is much less than official one

## Steps to reproduce

### Dataset
* Download & setup from [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)

### To check the generator network is properly copied
1. export pretrained weight from official code
2. set variable names to current implementation
3. try to generate from official weights `inference_from_official_weights.py`
4. check the result

| official output | current implementation |
|:---------------:|:----------------------:|
|![][official-output]|![][current-output]|

### Training
* Trying to use native tensorflow training framework ([tf.estimator])
* It recreates tf.estimator object every time it needs to change resolution or transition state is over to clear the optimizer states
* Run `train.py`
* training progress in tensorboard - 1024x1024
![][Training-progress-1024x1024]

### Generation results after training
* Generate using `inference.py`.

|  |  |
|:---------------:|:----------------------:|
|![][inference-0-1024x1024]|![][inference-1-1024x1024]|
|![][inference-2-1024x1024]|![][inference-3-1024x1024]|


[tf.contrib.distribute]: https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/MirroredStrategy
[tf.estimator]: https://www.tensorflow.org/guide/estimators
[official-output]: ./assets/example.png
[current-output]: ./assets/from-official-weights.png
[Training-progress-512x512]: ./assets/Training-progress-512x512.png
[Training-progress-1024x1024]: ./assets/Training-progress-1024x1024.png
[inference-0-1024x1024]: ./assets/inference-0-1024x1024.png
[inference-1-1024x1024]: ./assets/inference-1-1024x1024.png
[inference-2-1024x1024]: ./assets/inference-2-1024x1024.png
[inference-3-1024x1024]: ./assets/inference-3-1024x1024.png