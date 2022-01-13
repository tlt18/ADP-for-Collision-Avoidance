import numpy as np
import torch
from config import trainConfig

class Train():
    def __init__(self, env):
        self.env = env
        self.lossListValue = np.empty([0,1])
        self.lossListPolicy = np.empty([0,1])
        self.state = None
        config = trainConfig()
        self.stepForward = config.stepForward
        
    def reset(self):
        self.state = self.env.reset()

    def step(self, state, policy):
        control = policy.forward(self.state)
        self.state, _, done, _ = self.env.step(control)
        return done

    def policyEvaluate(self, policy, value):
        valuePredict = value(self.state)
        valueTarget = 0
        with torch.no_grad():
            stateNext = self.state
            for i in range(self.stepForward):
                control = policy.forward(stateNext)
                stateNext, reward, done, _ = self.env.Model(stateNext, control)
                valueTarget += reward
            valueTarget += (~done) * value(stateNext)
        lossValue = torch.pow(valuePredict - valueTarget, 2)
        value.zero_grad()
        lossValue.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        value.opt.step()
        value.scheduler.step()
        self.lossListValue = np.append(self.lossListValue, lossValue.detach().numpy())

    def policyImprove(self, policy, value):
        for p in value.parameters():
            p.requires_grad = False
        stateNext = self.state
        valueTarget = 0
        for i in range(self.stepForward):
            control = policy.forward(stateNext)
            stateNext, reward, done, _ = self.env.Model(stateNext, control)
            valueTarget += reward
        valueTarget += (~done) * value(stateNext)
        for p in value.parameters():
            p.requires_grad = True
        policy.zero_grad()
        lossPolicy = - valueTarget
        lossPolicy.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        policy.opt.step()
        policy.scheduler.step()
        self.lossListPolicy = np.append(self.lossListPolicy, lossPolicy.detach().numpy())