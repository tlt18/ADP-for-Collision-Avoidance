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
stateDim = env.observation_space.shape
actionDim = env.action_space.shape[0]
policy = Actor(stateDim, actionDim, config.lrPolicy)
value = Critic(stateDim, 1, config.lrValue)

if isTrain:
    print("Start Training!")

    iterationPEV = 1
    iterationPIM = 1
    train = Train(env)
    iterarion = 0
    lossListValue

    while iterarion < config.iterationMax:
        # train
        Train.reset()
        while True:
            # PEV
            if iterarion % iterationPEV == 0:
                Train.policyEvaluate(policy, value)
            # PIM
            if iterarion % iterationPIM == 0:
                Train.policyImprove(policy, value)
            done = Train.step(policy, value)
            if done:
                break
        if iterarion % config.iterationPrint:
            print("iteration: {}, LossValue: {}, LossPolicy: {}".format(iterarion, Train.lossListValue[-1], Train.lossListPolicy[-1]))
        if iterarion % config.iterationSave:
            pass
        iterarion += 1