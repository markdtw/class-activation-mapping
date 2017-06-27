# Class Activation Mapping
Tensorflow implementation of [Learning Deep Features for Discriminative Localization by Zhou et al](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) presented in CVPR'16.

Caffe version by the author is [here](https://github.com/metalbubble/CAM)

## Prerequisites
- Python 2.7+
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)
- [Caffe](https://github.com/bvlc/caffe) (only for extracting parameters from pre-trained model)
- [matplotlib](https://matplotlib.org/index.html) (for plotting the result)


## Data
- [CALTECH256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)


## Preparation
1. Clone this repo, create `log/` and `caffe_pretrained/` folder:
    ```bash
    git clone https://github.com/markdtw/class-activation-mapping.git
    cd class-activation-mapping
    mkdir caffe_pretrained
    mkdir log
    ```
2. To train on CALTECH256 dataset, download the original VGG16 graph (.prototxt) and model (.caffemodel) from [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8), save them in `caffe_pretrained/` folder.

3. To test directly from the pretrained ImageNet model, download the vgg16CAM graph and model from the [author's repo](https://github.com/metalbubble/CAM), save them in `caffe_pretrained/` folder as well.

4. If you went through both 2 and 3, your `caffe_pretrained/` folder should contain these:
    - `vgg16CAM_train_iter_90000.caffemodel`
    - `vgg16CAM_deploy.prototxt`
    - `VGG_ILSVRC_16_layers.caffemodel`
    - `VGG_ILSVRC_16_layers_deploy.prototxt`
    
    We need these only to convert them into .npy format.

5. Run `extract` function in `utils.py` with proper input arguments, this will convert .caffemodel to .npy. Now your `caffe_pretrained/` folder should have these two extra files:
    - `vgg16CAM_train_iter_90000.npy`
    - `VGG_ILSVRC_16_layers.npy`
    
    Let me know if you don't want to install caffe but still need them.


## Train
Train (fine-tune) CALTECH256 from `VGG_ILSVRC_16_layers` with default settings:
```bash
python main.py --train
```
Train (fine-tune) CALTECH256 from previous checkpoint:
```bash
python main.py --train --modelpath=log/vgg16CAM_calt256-X
```
Check out tunable arguments:
```bash
python main.py
```

## Test
Test the model provided by the authors (trained on ImageNet)
```bash
python main.py --test --imgpath=/path/to/img.jpg
```
Test the model trained on CALTECH256 by you given epoch X
```bash
python main.py --test --imgpath=/path/to/img.jpg --modelpath=log/vgg16CAM_calt256-X
```
This will save a result figure in this directory.


## Others
- First time training will generate `calt256_224_224.tfrecords` file to your `CALTECH256/` folder to load data in queue.
- Unfortunately training on CALTECH256 has not yet been successful/completed (super low accuracy). Please let me know if you can train the model with good result.
- Testing from the ImageNet model works perfectly with the same architecture.
- Issues are more than welcome!


## Resources
- [The project website](http://cnnlocalization.csail.mit.edu/)
- [The paper](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)
- [Author's repo](https://github.com/metalbubble/CAM)
- vgg implementation highly based on [this repo](https://github.com/machrisaa/tensorflow-vgg)
- how to load data in queue with tensorflow [here](https://github.com/markdtw/tensorflow-queue-example)(sorry for self-promote)

