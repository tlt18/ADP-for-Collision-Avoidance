from config import trainConfig
from MyEnv import MyEnv
from network import Actor, Critic
from train import Train
import numpy as np
import torch


# mode setting
isTrain = 1

# parameters setting
config = trainConfig()

# random seed
np.random.seed(0)
torch.manual_seed(0)


env = MyEnv()
env.seed(0)
policy = Actor(inputSize, Outputsize, config.lrPolicy)
value = Critic(inputSize, Outputsize, config.lrvalue)

if isTrain:
    print("Start Training!")
    iterationPrint = 10
    iterationSave = 200
    train = Train(env)
    iterarion = 0

    while iterarion < config.maxIteration:
        # train
        state = Train.reset()
        while True:
            [state, reward, done, _] = Train.step(state)
            # PEV


            while done:
                pass

            pass

        iterarion += 1
