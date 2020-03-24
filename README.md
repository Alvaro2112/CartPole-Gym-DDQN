# Solving CartPole Gym environment using DDQN

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

To run it you will need the following dependencies:

```
import gym
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
from matplotlib import pyplot as plt
import tensorflow as tf
import time
```

### Installing

To clone the repository execute the following command:

```
git clone git@github.com:Alvaro2112/Solving-CartPole-Gym-environment-using-DDQN.git
```

## Running the tests

To train the model you can simply run:

```
python CartPole_Q.py 
```

If you wish, you can play around with the Hyperparameters to try to better the convergence time.

### Results


![Rewards vs Episodes](/rewards.jpg)
![Rewards vs Episodes](/rewards.jpg)
![Rewards vs Episodes](/rewards.jpg)

## Built With

* [OpenAI Gym](https://gym.openai.com/) - Environmnet used to train the Agent

## Authors

* **Alvaro Caudéran**

## Acknowledgments

* Part of this code was taken from https://github.com/sanjitjain2/q-learning-for-cartpole/blob/master/qlearning.py

