# MNasNet
[Keras (Tensorflow)
Implementation](https://github.com/Shathe/MNasNet-Keras-Tensorflow/blob/master/Mnasnet.py)
of MNasNet and an example for training and evaluating it on the MNIST dataset.
Check also the [eager execution
implementation](https://github.com/Shathe/MNasNet-Keras-Tensorflow/blob/master/MnasnetEager.py)

According to the paper: [MnasNet: Platform-Aware Neural Architecture Search for
Mobile](https://arxiv.org/pdf/1807.11626.pdf)

## Requirement
* Python 3.5+
* Tensorflow-gpu 2.0 preview

## Train it
Train the [MNasNet
model](https://github.com/Shathe/MNasNet-Keras-Tensorflow/blob/master/Mnasnet.py)
on the ImageNet dataset! just execute:
```
./train.py
```
This assumes that ImageNet is under `~/datasets/imagenet`, see code for expected
paths and subdirectory names.

For checking and inspecting the Mnasnet model described in the paper, execute:
```
python3 Mnasnet.py
```



## Train it with eager execution
**NOTE:** eager doesn't work yet.
Train the [MNasNet (eager)
model](https://github.com/Shathe/MNasNet-Keras-Tensorflow/blob/master/MnasnetEager.py)
on the MNIST dataset! just execute:

```
python3 train_eager.py
```

The eager execution implementation also outputs logs on Tensorboard. For its visualization:
```
tensorboard --logdir=train_log:./logs/train, test_log:./logs/test
```

## MnasNet for... Semantic Segmentation!
In this other repository,
[FC-Mnasnet](https://github.com/Shathe/Semantic-Segmentation-Tensorflow-Eager)
I added a decoder to the MnasNet architecture in order to turn it into a
semantic segmentation model.



![alt text](https://github.com/Shathe/MNasNet-Keras-Tensorflow/raw/master/mnasnet.png)
