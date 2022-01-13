from config import trainConfig
from MyEnv import MyEnv
from network import Actor, Critic
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
    train = Train()
    iterarion = 0

    while iterarion < config.maxIteration:
        # train
        state = env.reset()
        while True:
            [state, reward, done, _] = env.step(state)



            pass
            

        iterarion += 1
