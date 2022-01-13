
class Train():
    def __init__(self, env, policy, value):
        self.env = env

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, state, policy):
        control = policy.forward(state)
        return env.step(control)
        