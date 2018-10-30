# Pseudo-Label The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks

The repository implement a semi-supervised method for Deep Neural Networks, the Pseudo Label. More details for the method please refer to *Pseudo-Label The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks*.

## The environment:

- Python 3.6.5 :: Anaconda
- PyTorch 0.4.0
- torchvision 0.2.1
- tensorboardX (for log)
- tensorflow (for visualization)

## To prepare the data:
```shell
bash data-local/bin/prepare_cifar10.sh
```

## To run the code:
```shell
python -m experiments.cifar10_test
```

## Visualization:
Make sure you have installed the tensorflow for tensorboard
```shell
tensorboard --logdir runs
```


## Code Reference

[pytorch-cifar@kuangliu](https://github.com/kuangliu/pytorch-cifar)

[mean-teacher@CuriousAI](https://github.com/CuriousAI/mean-teacher)

[senet.pytorch@moskomule](https://github.com/moskomule/senet.pytorch)

