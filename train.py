import numpy as np
import torch
from config import trainConfig


class Train():
    def __init__(self, env):
        self.env = env
        self.lossIteraValue = np.empty([0, 1])
        self.lossIteraPolicy = np.empty([0, 1])
        self.lossValue = []
        self.lossPolicy = []
        self.state = None
        self.reset()
        config = trainConfig()
        self.stepForward = config.stepForward
        self.batchSize = config.batchSize
        self.batchData = []

    def sample(self, policy):
        self.batchData = []
        for i in range(self.batchSize):
            self.batchData.append(self.state.clone())
            control = policy(self.state).detach()
            self.state, _, done, _ = self.env.step(control)
            if done:
                self.reset()
        self.batchData = torch.cat(
            self.batchData, dim=0).reshape(self.batchSize, -1)

    def reset(self):
        self.state = self.env.reset()

    def policyEvaluate(self, policy, value):
        valuePredict = value(self.batchData)
        valueTarget = torch.zeros(self.batchSize)
        with torch.no_grad():
            stateNext = self.batchData.clone()
            for i in range(self.stepForward):
                control = policy.forward(stateNext)
                stateNext, reward, done, _ = self.env.Model(stateNext, control)
                valueTarget += reward
            valueTarget += (~done) * value(stateNext)
        lossValue = torch.pow(valuePredict - valueTarget, 2).mean()
        value.zero_grad()
        lossValue.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        value.opt.step()
        value.scheduler.step()
        self.lossIteraValue = np.append(
            self.lossIteraValue, lossValue.detach().numpy())

    def policyImprove(self, policy, value):
        for p in value.parameters():
            p.requires_grad = False
        stateNext = self.batchData.clone()
        valueTarget = torch.zeros(self.batchSize)
        for i in range(self.stepForward):
            control = policy.forward(stateNext)
            stateNext, reward, done, _ = self.env.Model(stateNext, control)
            valueTarget += reward
        valueTarget += (~done) * value(stateNext)
        for p in value.parameters():
            p.requires_grad = True
        policy.zero_grad()
        lossPolicy = - valueTarget.mean()
        lossPolicy.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        policy.opt.step()
        policy.scheduler.step()
        self.lossIteraPolicy = np.append(
            self.lossIteraPolicy, lossPolicy.detach().numpy())

    def calLoss(self):
        self.lossValue.append(self.lossIteraValue.mean())
        self.lossPolicy.append(self.lossIteraPolicy.mean())
        self.lossIteraValue = np.empty([0, 1])
        self.lossIteraPolicy = np.empty([0, 1])
