from config import trainConfig
from MyEnv import MyEnv
from network import Actor, Critic
from train import Train
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime


# mode setting
isTrain = 1

# parameters setting
config = trainConfig()

# random seed
np.random.seed(0)
torch.manual_seed(0)

env = MyEnv()
env.seed(0)
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]
policy = Actor(stateDim, actionDim, config.lrPolicy)
value = Critic(stateDim, 1, config.lrValue)
log_dir = "./Results_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-")

if isTrain:
    print("Start Training!")

    iterationPEV = 1
    iterationPIM = 1
    train = Train(env)
    iterarion = 0
    lossListValue = 0

    while iterarion < config.iterationMax:
        # train
        train.reset()
        while True:
            # PEV
            if iterarion % iterationPEV == 0:
                train.policyEvaluate(policy, value)
            # PIM
            if iterarion % iterationPIM == 0:
                train.policyImprove(policy, value)
            done = train.step(policy, value)
            if done:
                train.calLoss()
                break
        if iterarion % config.iterationPrint == 0:
            print("iteration: {}, LossValue: {}, LossPolicy: {}".format(
                iterarion, train.lossValue[-1], train.lossPolicy[-1]))
        if iterarion % config.iterationSave == 0:
            pass
        iterarion += 1
    plt.figure()
    plt.plt(len(train.lossValue), train.lossValue)
    plt.xlabel('iteration')
    plt.ylabel('Value Loss')
    plt.savefig(log_dir + 'value_loss.png')
    plt.figure()
    plt.plt(len(train.lossPolicy), train.lossPolicy)
    plt.xlabel('iteration')
    plt.ylabel('Policy Loss')
    plt.savefig(log_dir + 'policy_loss.png')
