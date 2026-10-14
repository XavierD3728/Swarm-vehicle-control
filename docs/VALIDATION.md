# 验证范围与后续工作

## 自动测试

`uv run pytest -q` 覆盖：

- 恒定纵向加速度/角速度的解析轨迹与独立 ODE 积分；
- 零速输入映射和整个保持区间的轮速/线速度限制；
- 矩形平边的零切向曲率、角点区域 Hessian 的差分核对；
- 二维 QP 与独立 SLSQP 的随机多边形对照，不可行集合返回失败；
- 范围传感接口、延迟、丢包、缓存过期；
- 条件匹配、切换惩罚和身份保持；
- 强制求解失败的显式制动路径；
- 控制周期端点不碰撞但中间交叉时的审计；
- 混合式闭环输出和物理间隔记录。

## 结果应如何解释

应同时检查物理碰撞边界、请求 CBF 裕量、约束残差、降级事件、任务完成率和执行器限制。仅看到 GIF 没有相撞不能替代这些数据。

区间下界在静态地图、精确恒定输入动力学和理想本地感知的条件下成立。常数 cbf_margin 没有被当成针对任意采样和扰动的严格鲁棒界。出现 fallback 时，应查看 JSON events；即使最终物理轨迹安全，也不能声称该失败动作具有一般安全证书。

## 未实现为严格保证的部分

- 真实 ROS 2 / 多进程通信、传感器数据关联和 SLAM。
- 安全传感器噪声、遮挡、延迟、丢失；当前协同链路丢包不能被解释为安全链路丢包。
- 移动障碍速度及加速度、完整可达集预测。
- 任意拥堵状态下安全 QP 必然可行，或经证明的备用制动不变集。
- 时变有向图、切换匹配、导航与安全投影下全局能量单调或全局收敛。
- 全局概率密度的动态一致性估计；当前实施有限支撑局部拥挤。
- 相对于 V5 的完全同条件性能结论：V6 增加了边界安全、目标边界内缩、输入模型和切换惩罚。

## 文献依据

- Xiao & Belta, High-order control barrier functions: https://arxiv.org/abs/1903.04706
- Breeden, Garg & Panagou, sampled-data CBF: https://arxiv.org/abs/2103.03677
- Wang, Ames & Egerstedt, multirobot safety barriers: https://authors.library.caltech.edu/records/tshzw-g4v69
- Kia et al., dynamic average consensus: https://arxiv.org/abs/1803.04628

文献提供方法依据，不代表本项目自动满足其中全部假设。

