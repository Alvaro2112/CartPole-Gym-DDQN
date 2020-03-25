# Solving CartPole Gym environment using DDQN

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

To run it you will need the following dependencies:

```
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from matplotlib import pyplot as plt
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.optimizers import Adam
```

### Installing

To clone the repository execute the following command:

```
git clone git@github.com:Alvaro2112/Solving-CartPole-Gym-environment-using-DDQN.git
```

## Running the tests

To train the model you can simply run:

```
python CartPole_DQQ.py 
```

If you wish, you can play around with the Hyperparameters to try to better the convergence time.

### Results


![Rewards vs Episodes](/Reward_per_episode.jpg)

![Smoothed Rewards vs Episodes](/Smoothed_reward_per_episode.jpg)

![Epsilon decay](/Epsilon_value_decay.jpg)

Solving the environment may take more than 500 episodes sometimes.

## Built With

* [OpenAI Gym](https://gym.openai.com/) - Environmnet used to train the Agent

## Authors

* **Alvaro Caudéran**
