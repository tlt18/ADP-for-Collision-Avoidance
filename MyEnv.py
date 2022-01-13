# -*- coding: utf-8 -*-

"""
道阻且长，行则将至，行而不辍，未来可期。
"""

import torch
import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
import matplotlib.patches as mpaches
from matplotlib.pyplot import MultipleLocator

# 后加
import matplotlib.pyplot as plt
import time
from math import *


class MyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0):
        super(MyEnv, self).__init__()
        self.T = 0.1  # 时间间隔
        self.m = 1520  # 自车质量
        self.a = 1.19  # 质心到前轴的距离
        self.b = 1.46
        self.kf = -155495  # 前轮总侧偏刚度
        self.kr = -155495
        self.Iz = 2642  # 转动惯量

        # 计算车辆的外包络圆直径D
        self.L = 4.5  # 自车车长
        self.W = 1.85  # 自车宽度 单位：m
        self.t2 = []
        self.t1 = []
        self.T_count = 0

        # #######################################################################################################

        # 动作
        self.min_expect_acc = -4.0  # -2.0  # 期望的纵向加速度 单位：m/s2
        self.max_expect_acc = 0.0

        # 状态
        # 自车状态
        self.min_u_ego = 0  # 自车的纵向速度 单位：m/s  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.max_u_ego = 20 / 3.6  # 20/3.6

        # 后加纵向加速度和感知范围内的周车数目
        self.min_acc_x = -4.0  # -2.0  # 自车纵向加速度 单位：m/s2
        self.max_acc_x = 1.0

        # 周车状态
        self.min_D_long = -100.0  # 前车质心与自车质心的纵向相对距离  （车辆坐标系） 负号表示自车在前车之前
        self.max_D_long = 100.0  # 超过100m就变得无意义

        # 自车与前车的相对速度(前车速度减去自车速度) 单位：m/s （车辆坐标系） 负号：自车在前车之前
        self.min_u_rela = 0 / 3.6 - self.max_u_ego
        # 10/3.6               $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.max_u_rela = 10 / 3.6

        self.low_state = torch.tensor([self.min_u_ego, self.min_acc_x, self.min_D_long, self.min_u_rela],
                                  dtype=torch.float32)
        self.high_state = torch.tensor([self.max_u_ego, self.max_acc_x, self.max_D_long, self.max_u_rela],
                                   dtype=torch.float32)

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_expect_acc,
                                       high=self.max_expect_acc,
                                       shape=(1,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state.numpy(),
                                            high=self.high_state.numpy(),
                                            dtype=np.float32)

        # self.reset()

    def seed(self, seed):
        torch.manual_seed(seed)

    def reset(self):
        # 前车
        self.x_other = 150.0  # 初始化前车的x轴位置
        self.x_other_init = self.x_other
        # self.x_other = 60.0  # 初始化前车的x轴位置
        self.y_other = 0.0  # 初始化前车的y轴位置
        self.u_other = 0 / 3.6  # 初始化前车纵向速度                     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # 自车
        self.x_ego = np.random.uniform(low=0.0, high=149.5)  # 初始化自车x轴位置
        self.x_ego_init = self.x_ego
        # self.x_ego = 0.0  # 初始化自车x轴位置
        self.y_ego = 0.0  # 自车横向y轴位置
        self.u_ego_reset = self.max_u_ego  # 自车纵向速度
        self.v_ego = 0.0  # 自车横向速度
        self.phi_ego = 0.0  # 自车偏航角
        self.omega_ego = 0.0  # 自车横摆角速度
        # 初始化状态
        self.observation = torch.tensor([self.u_ego_reset,  # 自车纵向速度(车辆坐标系)
                                     0.0,  # 自车纵向加速度（车辆坐标系）
                                     self.x_other - self.x_ego,  # 纵距 （大地坐标系）
                                     self.u_other - self.u_ego_reset], dtype=torch.float32)  # 初始相对速度

        self.state = self.observation

        return (self.state - self.low_state )/ (self.high_state - self.low_state)

    def step(self, action, count_wzs=0):
        self.T_count = count_wzs
        # print(count_wzs)
        # =====================================自车状态转移========================================================
        # self.action = action[0]  # 自车纵向加速度
        self.action = action
        delta_ego = 0  # 自车前轮转角

        x_ego_0 = self.x_ego  # 自车纵向位置
        y_ego_0 = 0.0  # 自车的横向位置
        u_ego_0 = self.u_ego_reset  # 自车纵向速度
        v_ego_0 = 0.0  # 自车横向速度
        phi_ego_0 = 0.0  # 自车偏航角
        omega_ego_0 = 0.0  # 横摆角速度

        x_ego_1, y_ego_1, u_ego_1, v_ego_1, phi_ego_1, omega_ego_1 \
            = self.veh_dynamics(x_ego_0,
                                y_ego_0, u_ego_0, v_ego_0,
                                phi_ego_0, omega_ego_0,
                                self.action, delta_ego)
        # =======================================================================================================

        # *************************************前车状态转移********************************************************
        acc_other = 0.0
        delta_other = 0.0

        x_other_0 = self.x_other  # 前车纵向位置
        y_other_0 = 0  # 前车的横向位置
        u_other_0 = self.u_other  # 前车纵向速度
        v_other_0 = 0  # 前车横向速度
        phi_other_0 = 0  # 前车偏航角
        omega_other_0 = 0  # 前车的横摆角速度

        self.rela_distance = self.x_other - self.x_ego  # 画图用

        x_other_1, y_other_1, u_other_1, v_other_1, phi_other_1, omega_other_1 = self.veh_dynamics(x_other_0,
                                                                                                   y_other_0, u_other_0,
                                                                                                   v_other_0,
                                                                                                   phi_other_0,
                                                                                                   omega_other_0,
                                                                                                   acc_other,
                                                                                                   delta_other)
        # *******************************************************************************************************

        self.x_ego = x_ego_1  # 自车纵向位置
        u_ego_old = self.u_ego_reset
        # TODO: 原来clip的原理
        self.u_ego_reset = torch.clip(u_ego_1, 0, self.max_u_ego)  # 自车纵向速度
        u_ego_1 = self.u_ego_reset  # 这需要吗

        # 纵距、相对速度
        d_long = x_other_1 - x_ego_1  # 纵距
        d_long = torch.tensor([d_long], dtype = torch.float32)
        u_rela = u_other_1 - u_ego_1  # 纵向相对速度
        self.u_relate = u_rela

        # 前车
        self.x_other = x_other_1  # 前车纵向位置
        self.u_other = u_other_1  # 前车纵向速度

        # oooooooooooooooooooooooooooooooooooooo reward计算和state_next构建 oooooooooooooooooooooooooooooooooooooooooo
        done = False
        centroid_gap = self.L
        # 安全距离检测
        safe_gap = d_long - centroid_gap - 0.1

        # ============================================================================================
        # print(f'distance = {d_long:.4f}, velocity = {u_ego_1:.4f}, u_other = {u_other_1:.4f}, acc = {self.action:.4f}')
        # ============================================================================================

        # 首先判断是否碰撞
        if safe_gap <= 0.1:  # 有改动
            reward = torch.tensor([-80000],dtype = torch.float32) # TODO: reward的设计，不能求梯度
            done = True

            self.observation = torch.cat([u_ego_1, self.action, d_long, u_rela])
            self.state = self.observation
            return (self.state - self.low_state )/ (self.high_state - self.low_state), reward, done, {}

        # 自车速度为0后done掉
        if u_ego_1 <= 0.5:  # 有改动
            reward = torch.tensor([0],dtype = torch.float32)
            done = True

            #reward = 0
            # if 0.1 < d_long <= 2:
            #r_d = 10
            # elif 2 < d_long < 5:
            #r_d = 5
            # elif d_long >= 5:
            #r_d = -1

            # r_time = 20/count_wzs  # 时间越短奖励越大

            #reward = r_d + r_time

            self.observation = torch.cat([u_ego_1, self.action, d_long, u_rela])
            self.state = self.observation
            return (self.state - self.low_state )/ (self.high_state - self.low_state), reward, done, {}

        # 速度惩罚项
        # r_speed = 10 if abs(u_ego_1 - 50 / 3.6) < 0.1 or abs(u_ego_1 - 0 / 3.6) < 0.1 else -10

        # 距离惩罚项
        # r_distance = -10 * (d_long - 8) ** 2 + 20 if abs(d_long - 8)  < 1.0 else -10

        reward = 10 / ((u_ego_1 - 20 / 3.6) ** 2 + 0.1) - 1 * self.action ** 2
        #reward = 0
        self.observation = torch.cat([u_ego_1, self.action, d_long, u_rela])
        self.state = self.observation

        return (self.state - self.low_state )/ (self.high_state - self.low_state), reward, done, {}

        # ========================================================================================================

        # =========================================== 低速行驶 ====================================================
        """
        # 动作
        self.min_expect_acc = -4.0  # -2.0  # 期望的纵向加速度 单位：m/s2
        self.max_expect_acc = 0.0

        # 状态
        ## 自车状态
        self.min_u_ego = 0  # 自车的纵向速度 单位：m/s  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.max_u_ego = 45 / 3.6  # 20/3.6

        # 后加纵向加速度和感知范围内的周车数目
        self.min_acc_x = -4.0  # -2.0  # 自车纵向加速度 单位：m/s2
        self.max_acc_x = 1.0

        ## 周车状态
        self.min_D_long = -100.0  # 前车质心与自车质心的纵向相对距离  （车辆坐标系） 负号表示自车在前车之前
        self.max_D_long = 100.0  # 超过100m就变得无意义

        self.min_u_rela = 20 / 3.6 - self.max_u_ego  # 自车与前车的相对速度 单位：m/s （车辆坐标系） 负号：自车在前车之前
        self.max_u_rela = 10 / 3.6  # 10/3.6               $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        self.low_state = np.array([self.min_u_ego, self.min_acc_x, self.min_D_long, self.min_u_rela],
                                  dtype=np.float32)
        self.high_state = np.array([self.max_u_ego, self.max_acc_x, self.max_D_long, self.max_u_rela],
                                   dtype=np.float32)

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_expect_acc,
                                       high=self.max_expect_acc,
                                       shape=(1,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state,
                                            dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # 前车
        # self.x_other = 150.0  # 初始化前车的x轴位置
        self.x_other = 200.0  # 初始化前车的x轴位置
        self.y_other = 0.0  # 初始化前车的y轴位置
        self.u_other = 20 / 3.6  # 初始化前车纵向速度                     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # 自车
        # self.x_ego = np.random.uniform(low=0.0, high=149.5)  # 初始化自车x轴位置
        self.x_ego = 0.0  # 初始化自车x轴位置
        self.y_ego = 0.0  # 自车横向y轴位置
        self.u_ego_reset = self.max_u_ego  # 自车纵向速度
        self.v_ego = 0.0  # 自车横向速度
        self.phi_ego = 0.0  # 自车偏航角
        self.omega_ego = 0.0  # 自车横摆角速度
        # 初始化状态
        self.observation = np.array([self.u_ego_reset,  # 自车纵向速度(车辆坐标系)
                                     0.0,  # 自车纵向加速度（车辆坐标系）
                                     self.x_other - self.x_ego,  # 纵距 （大地坐标系）
                                     self.u_other - self.u_ego_reset], dtype=np.float32)  # 初始相对速度

        self.state = self.observation

        return self.state / self.high_state

    def step(self, action, count_wzs):
        self.T_count = count_wzs
        # =====================================自车状态转移========================================================
        self.action = action[0]  # 自车纵向加速度
        delta_ego = 0  # 自车前轮转角

        x_ego_0 = self.x_ego  # 自车纵向位置
        y_ego_0 = 0.0  # 自车的横向位置
        u_ego_0 = self.u_ego_reset  # 自车纵向速度
        v_ego_0 = 0.0  # 自车横向速度
        phi_ego_0 = 0.0  # 自车偏航角
        omega_ego_0 = 0.0  # 横摆角速度

        x_ego_1, y_ego_1, u_ego_1, v_ego_1, phi_ego_1, omega_ego_1 = self.veh_dynamics(x_ego_0,
                                                                                       y_ego_0, u_ego_0, v_ego_0,
                                                                                       phi_ego_0, omega_ego_0,
                                                                                       self.action, delta_ego)
        # =======================================================================================================

        # *************************************前车状态转移********************************************************
        acc_other = 0.0
        delta_other = 0.0

        x_other_0 = self.x_other  # 前车纵向位置
        y_other_0 = 0  # 前车的横向位置
        u_other_0 = self.u_other  # 前车纵向速度
        v_other_0 = 0  # 前车横向速度
        phi_other_0 = 0  # 前车偏航角
        omega_other_0 = 0  # 前车的横摆角速度

        self.rela_distance = self.x_other - self.x_ego  # 画图用


        x_other_1, y_other_1, u_other_1, v_other_1, phi_other_1, omega_other_1 = self.veh_dynamics(x_other_0,
                                                                                                   y_other_0, u_other_0,
                                                                                                   v_other_0,
                                                                                                   phi_other_0,
                                                                                                   omega_other_0,
                                                                                                   acc_other,
                                                                                                   delta_other)
        # *******************************************************************************************************

        # 纵距、相对速度
        d_long = x_other_1 - x_ego_1  # 纵距
        u_rela = u_other_1 - u_ego_1  # 纵向相对速度
        self.u_relate = u_rela

        self.x_ego = x_ego_1  # 自车纵向位置
        u_ego_old = self.u_ego_reset

        self.u_ego_reset = np.clip(u_ego_1, 0, self.max_u_ego)  # 自车纵向速度

        # 前车
        self.x_other = x_other_1  # 前车纵向位置
        self.u_other = u_other_1  # 前车纵向速度

        # oooooooooooooooooooooooooooooooooooooo reward计算和state_next构建 oooooooooooooooooooooooooooooooooooooooooo
        done = False
        centroid_gap = self.L
        # 安全距离检测
        safe_gap = d_long - centroid_gap - 0.1

        # ============================================================================================
        # print(f'distance = {d_long:.4f}, velocity = {u_ego_1:.4f}, u_other = {u_other_1:.4f}, acc = {self.action:.4f}')
        # ============================================================================================

        # 首先判断是否碰撞
        if safe_gap <= 0.1:  # 有改动
            reward = -80000
            done = True

            self.observation = np.array([u_ego_1, self.action, d_long, u_rela])
            self.state = self.observation
            return self.state / self.high_state, reward, done, {}

        # 自车速度小于前车速度时done掉
        if u_ego_1 <= 20/3.6:  # 有改动
            reward = 0
            done = True

            self.observation = np.array([u_ego_1, self.action, d_long, u_rela])
            self.state = self.observation
            return self.state / self.high_state, reward, done, {}

        # 速度惩罚项
        # r_speed = 10 if abs(u_ego_1 - 50 / 3.6) < 0.1 or abs(u_ego_1 - 0 / 3.6) < 0.1 else -10

        # 距离惩罚项
        # r_distance = -10 * (d_long - 8) ** 2 + 20 if abs(d_long - 8)  < 1.0 else -10

        reward = 10 / ((u_ego_1 - 45 / 3.6) ** 2 + 0.1) - 1 * self.action ** 2

        self.observation = np.array([u_ego_1, self.action, d_long, u_rela])
        self.state = self.observation

        return self.state / self.high_state, reward, done, {}
        """

        # ===================================================================================================

        # =================================================== 前车制动 ==========================================================
        """                                                        前车制动
        # 动作
        self.min_expect_acc = -5.0  # -2.0  # 期望的纵向加速度 单位：m/s2
        self.max_expect_acc = 0.0

        # 状态
        ## 自车状态
        self.min_u_ego = 0  # 自车的纵向速度 单位：m/s  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.max_u_ego = 50 / 3.6  # 20/3.6

        # 后加纵向加速度和感知范围内的周车数目
        self.min_acc_x = -5.0  # -2.0  # 自车纵向加速度 单位：m/s2
        self.max_acc_x = 1.0

        ## 周车状态
        self.min_D_long = -100.0  # 前车质心与自车质心的纵向相对距离  （车辆坐标系） 负号表示自车在前车之前
        self.max_D_long = 100.0  # 超过100m就变得无意义


        self.min_u_rela = 0/3.6 - self.max_u_ego   # 自车与前车的相对速度 单位：m/s （车辆坐标系） 负号：自车在前车之前
        self.max_u_rela = 10/3.6   # 10/3.6               $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


        self.low_state = np.array([self.min_u_ego, self.min_acc_x, self.min_D_long,  self.min_u_rela],
                                  dtype=np.float32)
        self.high_state = np.array([self.max_u_ego, self.max_acc_x, self.max_D_long, self.max_u_rela],
                                   dtype=np.float32)

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_expect_acc,
                                       high=self.max_expect_acc,
                                       shape=(1,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state,
                                            dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        # 前车
        # self.x_other = 150.0  # 初始化前车的x轴位置
        self.x_other = 40.0  # 初始化前车的x轴位置
        self.y_other = 0.0  # 初始化前车的y轴位置
        self.u_other = 50 / 3.6 # 初始化前车纵向速度                     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # 自车
        # self.x_ego = np.random.uniform(low=0.0, high=149.5)  # 初始化自车x轴位置
        self.x_ego = 0.0  # 初始化自车x轴位置
        self.y_ego = 0.0  # 自车横向y轴位置
        self.u_ego_reset = self.max_u_ego  # 自车纵向速度
        self.v_ego = 0.0  # 自车横向速度
        self.phi_ego = 0.0  # 自车偏航角
        self.omega_ego = 0.0  # 自车横摆角速度
        # 初始化状态
        self.observation = np.array([self.u_ego_reset,  # 自车纵向速度(车辆坐标系)
                               0.0,  # 自车纵向加速度（车辆坐标系）
                               self.x_other - self.x_ego,  # 纵距 （大地坐标系）
                               self.u_other - self.u_ego_reset], dtype=np.float32)  # 初始相对速度

        self.state = self.observation

        return self.state/self.high_state


    def step(self, action, count_wzs):
        self.T_count = count_wzs
        # =====================================自车状态转移========================================================
        self.action = action[0]  # 自车纵向加速度
        delta_ego = 0  # 自车前轮转角

        x_ego_0 = self.x_ego  # 自车纵向位置
        y_ego_0 = 0.0  # 自车的横向位置
        u_ego_0 = self.u_ego_reset  # 自车纵向速度
        v_ego_0 = 0.0  # 自车横向速度
        phi_ego_0 = 0.0  # 自车偏航角
        omega_ego_0 = 0.0  # 横摆角速度

        x_ego_1, y_ego_1, u_ego_1, v_ego_1, phi_ego_1, omega_ego_1 = self.veh_dynamics(x_ego_0,
                                    y_ego_0,u_ego_0, v_ego_0, phi_ego_0, omega_ego_0, self.action, delta_ego)
        # =======================================================================================================

        # *************************************前车状态转移********************************************************
        acc_other = 0.0 if count_wzs < 100 else -4.0
        delta_other = 0.0

        x_other_0 = self.x_other  # 前车纵向位置
        y_other_0 = 0  # 前车的横向位置
        u_other_0 = self.u_other  # 前车纵向速度
        v_other_0 = 0  # 前车横向速度
        phi_other_0 = 0  # 前车偏航角
        omega_other_0 = 0  # 前车的横摆角速度

        x_other_1, y_other_1, u_other_1, v_other_1, phi_other_1, omega_other_1 = self.veh_dynamics(x_other_0,
                            y_other_0, u_other_0, v_other_0, phi_other_0, omega_other_0, acc_other, delta_other)
        # *******************************************************************************************************

        # 纵距、相对速度
        d_long = x_other_1 - x_ego_1  # 纵距
        u_rela = u_other_1 - u_ego_1  # 纵向相对速度
        self.u_relate = u_rela

        self.x_ego =  x_ego_1  # 自车纵向位置
        u_ego_old = self.u_ego_reset

        self.u_ego_reset = np.clip(u_ego_1, 0, self.max_u_ego)  # 自车纵向速度

        # 前车
        self.x_other =  x_other_1  # 自车纵向位置
        u_other_1 = 0.0 if u_other_1 <=0 else u_other_1
        self.u_other = u_other_1  # 自车纵向速度

        # oooooooooooooooooooooooooooooooooooooo reward计算和state_next构建 oooooooooooooooooooooooooooooooooooooooooo
        done = False
        centroid_gap = self.L
        # 安全距离检测
        safe_gap = d_long - centroid_gap - 0.1

        # ============================================================================================
        #print(f'distance = {d_long:.4f}, velocity = {u_ego_1:.4f}, u_other = {u_other_1:.4f}, acc = {self.action:.4f}')
        # ============================================================================================

        # 首先判断是否碰撞
        if safe_gap <= 0.1:  # 有改动
            reward = -80000
            done = True

            self.observation = np.array([u_ego_1, self.action, d_long, u_rela])
            self.state = self.observation
            return self.state/self.high_state, reward, done, {}

        # 自车速度为0后done掉
        if u_ego_1 <= 0.5:  # 有改动
            reward = 0
            done = True

            self.observation = np.array([u_ego_1, self.action, d_long, u_rela])
            self.state = self.observation
            return self.state/self.high_state, reward, done, {}

        # 速度惩罚项
        #r_speed = 10 if abs(u_ego_1 - 50 / 3.6) < 0.1 or abs(u_ego_1 - 0 / 3.6) < 0.1 else -10

        # 距离惩罚项
        #r_distance = -10 * (d_long - 8) ** 2 + 20 if abs(d_long - 8)  < 1.0 else -10

        reward = 10/((u_ego_1 - 50 / 3.6)**2 + 0.1) - 1 * self.action**2

        self.observation = np.array([u_ego_1, self.action, d_long, u_rela])
        self.state = self.observation

        return self.state / self.high_state, reward, done, {}
        """

    def veh_dynamics(self, x_0, y_0, u_0, v_0, phi_0, omega_0, action, delta):
        x_1 = x_0 + self.T * (u_0 * np.cos(phi_0) - v_0 * np.sin(phi_0))
        y_1 = y_0 + self.T * (v_0 * np.cos(phi_0) + u_0 * np.sin(phi_0))
        u_1 = u_0 + self.T * action
        v_1 = (-(self.a * self.kf - self.b * self.kr) * omega_0 + self.kf * delta * u_0 +
               self.m * omega_0 * u_0 * u_0 - self.m * u_0 * v_0 / self.T) / (self.kf +
                                                                              self.kr - self.m * u_0 / self.T)
        phi_1 = phi_0 + self.T * omega_0
        omega_1 = (-self.Iz * omega_0 * u_0 / self.T - (self.a * self.kf - self.b * self.kr) * v_0
                   + self.a * self.kf * delta * u_0) / ((self.a * self.a * self.kf + self.b *
                                                         self.b * self.kr) - self.Iz * u_0 / self.T)
        return x_1, y_1, u_1, v_1, phi_1, omega_1

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(60, 40)

        plt.ion()  # 开启interactive mode 成功的关键函数

        """=========================================== 源代码 ==============================================="""
        # fig=plt.figure(6)
        # plt.plot([-10, 150], [5.5, 5.5], c='b', linestyle='--',linewidth=0.6)
        # plt.plot([-10, 150], [-5.5, -5.5], c='b', linestyle='--',linewidth=0.6)
        # plt.plot(self.x_other, self.y_other, '*')
        # plt.plot(self.x_ego, self.y_ego, '.')  # 第次对画布添加一个点，覆盖式的。
        # plt.axis([-10, 150, -50, 50])
        #
        #
        # # plot car_other
        # ax = plt.gca()
        # rect=plt.Rectangle((self.x_other-4.5/2, self.y_other-2/2), 4.5, 2,angle=0)
        # ax.add_patch(rect)
        #
        # # # plot car_ego
        # ax1 = plt.gca()
        # rect1 = plt.Rectangle((self.x_ego - 4.5 / 2, self.y_ego - 2 / 2), 4.5, 2,angle=0)
        # ax1.add_patch(rect1)
        #
        # # plt.show()
        # plt.xlabel('X/m')
        # plt.ylabel('Y/m')
        # plt.title('Movement track of ego vehicle')
        # plt.pause(0.1)
        # plt.clf()
        """==================================================================================================="""

        """=========================================== 自车速度 ==============================================="""
        fig = plt.figure(1)
        plt.plot(self.T_count, self.u_ego_reset,
                 '.', c='b')  # 第次对画布添加一个点，覆盖式的。
        # plt.plot(self.T_count, self.u_other, '.', c='r')  # 第次对画布添加一个点，覆盖式的。
        # plt.plot(self.T_count, self.u_relate, '.', c='g')  # 第次对画布添加一个点，覆盖式的。
        plt.axis([-10, 600, -15, 15])

        plt.xlabel('t(0.1s)')
        plt.ylabel('v(m/s)')
        plt.title('AEB')
        # plt.pause(0.1)
        # plt.pause(0)
        # plt.show()
        """=================================================================================================="""

        """=========================================== 相对距离 ==============================================="""
        fig = plt.figure(2)
        plt.plot(self.T_count, self.rela_distance, '.')  # 第次对画布添加一个点，覆盖式的。
        plt.axis([-10, 600, -5, 210])

        plt.xlabel('t(0.1s)')
        plt.ylabel('Distance(m)')
        plt.title('AEB')
        # plt.pause(0.1)
        # plt.show()
        """=================================================================================================="""

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def Model(self, state, action):
        # update ego state
        state_ = state * ( self.high_state - self.low_state ) + self.low_state

        delta_ego = 0  # 自车前轮转角
        x_ego_0 = 0  # 自车纵向位置，前车纵向距离直接取为两者距离差
        y_ego_0 = 0.0  # 自车的横向位置
        u_ego_0 = state_[0]  # 自车纵向速度
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
        x_other_0 = state_[2]  # 前车纵向位置
        y_other_0 = 0  # 前车的横向位置
        u_other_0 = state_[3] + state_[0]  # 前车纵向速度
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
        d_long = torch.tensor([d_long],dtype = torch.float32)
        u_rela = u_other_1 - u_ego_1  # 纵向相对速度

        # calculate reward
        done = False
        centroid_gap = self.L
        safe_gap = d_long - centroid_gap - 0.1

        # collision case
        if safe_gap <= 0.1:  # 有改动
            reward = torch.tensor([-80000],dtype = torch.float32)
            done = True
            stateUpd = torch.cat([u_ego_1, action, d_long, u_rela])
            return stateUpd / self.high_state, reward, done, {}

        # stop case
        if u_ego_1 <= 0.5:  # 有改动
            reward = torch.tensor([0],dtype = torch.float32)
            done = True
            stateUpd = torch.cat([u_ego_1, action, d_long, u_rela])
            return stateUpd / self.high_state, reward, done, {}

        # normal case
        reward = 10 / ((u_ego_1 - 20 / 3.6) ** 2 + 0.1) - 1 * action ** 2
        
        stateUpd = torch.cat([u_ego_1, action, d_long, u_rela])
        return (stateUpd - self.low_state )/ (self.high_state - self.low_state), reward, done, {}


def test_env():
    env = MyEnv()
    state = env.reset()
    print('init state', state)
    time.sleep(2)
    T_count = 0
    while True:
        [state, reward, done, _] = env.step(np.array([-math.pi/60]), T_count)
        env.render()
        T_count += 1
        # time.sleep(0.05)
        print('state', state)
        print('reward', reward)
        print('done', done)


if __name__ == "__main__":
    test_env()
