# mlAlgorithms
This repo contains some basic ML models like Alexnet,VGG,Resnet...<br/>
Package tensorflow contains these models implementations using tensorflow >= 1.0.<br/>
Package mxnet contains these models implementations using mxnet >= 0.10.

Indeed,all the models can be found at model zoom of relative frameworks like tensorflow,mxnet

### Warn
I have worked with mxnet,so I can maintain the package mxnet continually.Models at package 
mxnet are migrated from [mxnet symbols](https://github.com/apache/incubator-mxnet/tree/master/example) 
and [mxnet model zoom](http://data.mxnet.io/models/).

## Installation
Requirements:
* python3
* tensorflow >= 1.0 for package tensorflow
* mxnet >= 0.10 for package mxnet
* opencv3
* yaml
* numpy
* matplotlib
* collections
* PIL(pillow)

## Structure
- mxnet
    - cv_tools (*tools process image and video using opencv*)
    - imagenet (*trained weights on imagenet-1000 from mxnet model zoo, .json file and .params file*)
    - model (*model from mxnet model zoom,you can add new models to this package as you need*)
    - tools (*some useful tools,train log visualized tool for now*)
    - trainedmodel (*trained model on your own dataset*)

## Usage

### train on your dataset

```bash
python train.py --parampath params.yaml
```
All the parameters needed for training configured in params.yaml.

### test with your model
```bash
python batch_eval.py \
    --imagepath /home/gy/gitpro/mlAlgorithms/mxnet/cv_tools/image \
    --modelprefix trainedmodel/model \
    --epoch 2
```
label.txt is a file contains <br/>
<b>classification index---classification name</b> pair

### process your dataset
At package cv_tools
```bash
ptrhon pack.py \
    --imagepath iamges \
    --datapath traindata \
    --imgsize "499,499" \
    --channel 3 \
    --slice 1
```


## Todo
`- [ ]` add mxnet Rec format <br/>
`- [ ]` using gluon <br/>
`- [ ]` multiple gpus <br/>