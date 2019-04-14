# Implementation details

## Progressive training
* From the paper [PROGRESSIVE GROWING OF GANS]
    - All existing layers in both networks remain trainable throughout the training process.
    - ![][PGGAN-fig01]
    - When new layers are added to the networks, we fade them in smoothly.
    - ![][PGGAN-fig02]
        + alpha is flipped in official code
        + `implementation_alpha = (1.0 - paper_alpha)`

## Normalization techniques
* From the paper [PROGRESSIVE GROWING OF GANS]
    - Equalized learning rate
        + initialize weights via `N(0, 1)` but scales with per-layer normalization constant from He’s initializer
            * `w^_i=w_i/c`
            * `c= gain * sqrt(1.0/number_of_inputs)`

[Progressive Growing of GANs]: https://arxiv.org/abs/1710.10196
[PGGAN-fig01]: ./assets/PGGAN-fig01.png
[PGGAN-fig02]: ./assets/PGGAN-fig02.png