# necto-nn
An implementaion of feed forward neural network with different optimizers from scratch.

This is a simple deep learning library created for learning deep learning concepts by implementing them from scratch using numpy. The structure and algorithms are inspired by popular frameworks such as tensorflow, pytorch and huggingface trainers.

The repo was make like a python package which contains multiple python files each for different components requred to train a feedforward neural network.

Checkout [demo.ipynb](demo.ipynb) and [train.py](train.py) for the example usage of the API.


- [Github Repo](https://github.com/livinNector/necto-nn)
- [Assignment Report](https://wandb.ai/livinNector-academic/deep-learning-course/reports/Livin-Nector-s-DA6401-Assignment-1--VmlldzoxMTY4NTc0NA)

- [x] `nn` - The feedforward neural network.
- [x] `initializers`
  - [x] RandomInit
  - [x] XavierInit
- [x] `activations`
  - [x] Identity
  - [x] Sigmoid
  - [x] TanH
  - [x] ReLU
  - [x] SoftMax
- [x] `losses`
  - [x] MSELoss
  - [x] CrossEntropyLoss (one hot encoded labels)
  - [x] SparseCrossEntropyLoss (numeric labels)
- [x] `metrics`
  - [x] accuracy
  - [x] f1_score
- [x] `optimizers`
  - [x] SGD
  - [x] Momentum
  - [x] NAG
  - [x] RMSProp
  - [x] Adam
  - [x] NAdam
- [x] `trainer` - Huggingface Trainer like class for training the NN.
- [x] `utils` 


