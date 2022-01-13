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
value = Critic(inputSize, Outputsize, config.lrValue)

if isTrain:
    print("Start Training!")
    iterationPrint = 10
    iterationSave = 200
    iterationPEV = 1
    iterationPIM = 1
    train = Train(env)
    iterarion = 0

    while iterarion < config.maxIteration:
        # train
        Train.reset()
        while True:
            # PEV
            if iterarion % iterationPEV == 0:
                [lossListValue, done] = Train.policyEvaluate(policy, value)
            # PIM
            if iterarion % iterationPIM == 0:
                Train.policyImprove(policy, value)


            while done:
                pass

            pass

        iterarion += 1
