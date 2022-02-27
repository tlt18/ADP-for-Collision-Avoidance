# ADP-for-Collision-Avoidance
Approximate dynamic programming for collision avoidance

### 说明
这是最初的毕设题目，使用ADP车辆避撞。
设想将避撞转化为IDC中静态路径选择、动态路径跟踪问题，因此老师直接将毕设题目换成了ADP做高精度的轨迹跟踪。
毕设从一个IDC的应用问题到IDC其中一个模块的改进问题。

### 问题
环境目前只考虑了纵向的运动，即两个积分器。由于奖励设置过于稀疏，因此用现有算法ADP不能跑通。
