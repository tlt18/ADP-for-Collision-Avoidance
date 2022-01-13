import numpy as np
import pytorch

class Train():
    def __init__(self, env):
        self.env = env
        self.lossListValue = np.empty([0,1]) 
        self.state = None
        self.stateNest = None

    def reset(self):
        self.state = self.env.reset()

    def step(self, state, policy):
        control = policy.forward(self.state)
        return self.env.step(control)

    def policyEvaluate(self, policy, value):
        control = policy.forward(self.state)
        # [stateNext, reward, done, _] = self.env.Model(state, control)
        [stateNext, reward, done, _] = self.env.step(control)
        lossValue = torch.pow(value.forward(self.state) - reward.detach() - value.forward(stateNext).detach(), 2)
        value.zero_grad()
        lossValue.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        value.opt.step()
        value.scheduler.step()
        self.lossListValue = np.append(self.lossListValue, lossValue.detach().numpy())
        self.reward = reward

        return self.lossListValue[-1], done

    def policyImprove(self, policy, value):
        [stateNext, reward, done, _] = self.env.model(control)

        pass

        