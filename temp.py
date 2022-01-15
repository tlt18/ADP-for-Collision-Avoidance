def Model(self, state, action):
    # update ego state
    state_ = state * ( self.high_state - self.low_state ) + self.low_state

    delta_ego = 0  # 自车前轮转角
    x_ego_0 = 0  # 自车纵向位置，前车纵向距离直接取为两者距离差
    y_ego_0 = 0.0  # 自车的横向位置
    u_ego_0 = state_[:, 0]  # 自车纵向速度
    v_ego_0 = 0.0  # 自车横向速度
    phi_ego_0 = 0.0  # 自车偏航角
    omega_ego_0 = 0.0  # 横摆角速度
    x_ego_1, y_ego_1, u_ego_1, v_ego_1, phi_ego_1, omega_ego_1 \
        = self.veh_dynamics(x_ego_0,\
                            y_ego_0, u_ego_0, v_ego_0,\
                            phi_ego_0, omega_ego_0,\
                            action, delta_ego)
    u_ego_1 = torch.clip(u_ego_1, 0, self.max_u_ego) # 这里有点问题
    # update other state
    acc_other = 0.0
    delta_other = 0.0
    x_other_0 = state_[:,2]  # 前车纵向位置
    y_other_0 = 0  # 前车的横向位置
    u_other_0 = state_[:,3] + state_[:,0]  # 前车纵向速度
    v_other_0 = 0  # 前车横向速度
    phi_other_0 = 0  # 前车偏航角
    omega_other_0 = 0  # 前车的横摆角速度
    x_other_1, y_other_1, u_other_1, v_other_1, phi_other_1, omega_other_1 \
        = self.veh_dynamics(x_other_0,\
                            y_other_0, u_other_0, v_other_0,\
                            phi_other_0, omega_other_0,\
                            acc_other, delta_other)
    # output state
    d_long = x_other_1 - x_ego_1  # 纵距
    u_rela = u_other_1 - u_ego_1  # 纵向相对速度

    # calculate reward
    centroid_gap = self.L
    safe_gap = d_long - centroid_gap - 0.1
    # normal case
    done = torch.tensor([False for i in range(state.size(0))])
    reward = torch.tensor([0 for i in range(state.size(0))])
    reward = 10 / ((u_ego_1 - 20 / 3.6) ** 2 + 0.1) - 1 * action ** 2 # TODO: 对action惩罚要大一点
    # stop case
    indexStop = u_ego_1 <= 0.5
    done[indexStop] = True
    reward[indexStop] = - u_ego_1[indexStop]**2/10
    # collision case
    indexCollision = safe_gap <= 0.1
    done[indexCollision] = True
    reward[indexCollision] = 1000 * (safe_gap[indexCollision] - 0.1) - 300

    stateUpd = torch.stack([u_ego_1, action, d_long, u_rela],dim = 1).reshape(state.size(0), -1)
    return (stateUpd - self.low_state )/ (self.high_state - self.low_state), reward, done, {}




