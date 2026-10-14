# V6 数学模型与模块对应

本文件描述实际闭环，而不是将所有模块都解释为同一个光滑梯度流。

## 1. 群体、角色与目标

机器人状态为 z_i=(p_i,theta_i,s_i)，角色 r_i 和能力 c_i 固定。能力向量包含载荷、最大速度/1.4、电池和冷链传感能力。角色条件经验分布为：

```math
mu_r = (1/N_r) sum_{i:r_i=r} delta_{p_i},
nu_r = (1/N_r) sum_{j:zeta_j=r} delta_{y_j}.
```

能力核还需要标记经验分布 eta=(1/N) sum_i delta_(c_i,r_i,p_i)。这些分布是粒子数组的数学表示，机器人没有通过全局数组获得完整概率密度。

V5 两个跨角色目标锚点只相距约 0.049 m，小于机器人直径。V6 在初始化时于原角色区域内、原锚点附近构造固定槽位，最小间距为 max(D+0.12,0.68) m，并与墙体保持裕量。运行过程中不通过移动官方目标或投影机器人位置修饰结果。

对应：`scenario.py:Scenario, build_scenario`。

## 2. 中心低频分配与标签交换

```math
sigma_m = argmin_sigma sum_i [||p_i-y_sigma(i)||^2
                              + kappa_switch 1(sigma(i) != sigma_previous(i))],
subject to sigma being a bijection and zeta_sigma(i)=r_i.
```

更新之间保持 sigma。仅在映射实际变化时更新分配版本。交换的是目标置换，不是机器人 ID、能力或角色。中心发送目标，不发送机器人完整避障轨迹。

对应：`assignment.py:assign`、`experiment.py:run`。

## 3. 本地观测、消息与位置估计

安全视图为 LocalView(own, observed, messages)。真实世界适配器执行范围查询，机器人控制器只接收局部视图。安全传感器是理想、同步、无遮挡、无噪声的范围传感器；协同消息独立模拟逐接收者丢包、延迟、缓存过期。

导航维护每台机器人自己的观测记忆：
```math
hat p_ij(t) = p_j(t_last) + (t-t_last) v_j(t_last)
```
运动邻居只短时预测 0.5 s；速度低于 0.08 m/s 的停车邻居位置保留最多 30 s，避免一离开视野便忘记被堵通道。新观测覆盖旧记录。

这是一种本地有限记忆位置估计，不是全局分布一致性估计，也不是安全状态的精确替代。CBF 使用未经这个记忆扩展的实时观测，导航使用预测视图。

局部 KDE 为传感范围内高斯核强度，不归一化成全局概率分布。

对应：`communication.py`、`local_planner.py:planning_view`、`guidance.py:local_density`。

## 4. 导航、局部路径与协同

静态地图按欧氏距离膨胀，八邻域最短路禁止对角切角。每台机器人独立在静态先验上叠加本地观测/记忆中的机器人缓冲圆，计算到自己当前分配目标的路径，每 0.5 s 或目标变化时更新。连续起点通过不降低当前净空的线段连接自由网格，不改变物理状态。

接近目标的通道可能被已到位机器人占据。消息中的 blocked/yielding 用于局部临时让行：已到位机器人收到请求后选取可通行的让行点，保持 3.5 s 后恢复原任务；让行点不修改 sigma 或官方目标。yielding 标记避免把让行位移当成新的任务请求。

能力交互核与误差反馈：
```math
W(c_i,c_j)=0.15+0.85 exp(-||c_i-c_j||^2/0.38),
e_i=p_i-y_sigma(i),
u_G,i=-(beta/N) sum_{j in N_coord,i} W(c_i,c_j)(e_i-hat e_ij).
```
协同只使用同角色、同分配版本、未过期的收到消息；图是有向且时变的，不强行假设瞬时对称。

有限支撑拥挤势：
```math
psi_R(d)=(1-d^2/R^2)^3 for d<R; 0 otherwise,
u_C,i=-gamma sum_j grad_i psi_R(d_ij)
     =(6 gamma/R^2) sum_{d_ij<R}(1-d_ij^2/R^2)^2(p_i-p_j).
```
它只使用局部传感，和 V5 全局高斯核不是同一能量。保留固定局部作用强度，不再使用 V5 拥挤项的 1/N 归一化。

实际期望速度：
```math
u_i = Sat_u_max(u_nav,i + u_G,i + u_C,i + u_obstacle,i).
```
u_nav 来源于机器人本地路径或暂时让行路径。单元接口没有传入 planner 时保留短距离几何绕行备选。

对应：`navigation.py`、`local_planner.py`、`guidance.py`。

## 5. 梯度流对应的基础能量

固定匹配和对称固定图时，目标与协同核心可写成：
```math
E_T = alpha/(2N) sum_i ||e_i||^2,
E_G = beta/(4N^2) sum_ij W_ij ||e_i-e_j||^2,
E_C = gamma/(2N) sum_{i != j} psi_R(d_ij).
u_base,i = -N grad_i(E_T+E_G+E_C).
```

在即时最优条件匹配处，E_T 等于角色加权 Wasserstein 平方距离的 alpha/2 倍。缓存或带切换惩罚的匹配不一定实现当时的纯运输最小值。

实际系统包含导航路径、临时让行、消息预测、有向时变图、输入饱和和安全优化，不宣称整个闭环是单个固定光滑能量的纯梯度流，也不宣称全局能量单调或任意场景无死锁。

## 6. 可执行差速动力学

```math
dot p_i=s_i t_i, dot theta_i=omega_i, dot s_i=b_i,
t_i=(cos theta_i,sin theta_i), n_i=(-sin theta_i,cos theta_i),
ddot p_i=B_i q_i, B_i=[t_i,s_i n_i], q_i=(b_i,omega_i).
```

安全层直接优化 q_i。零速时不除以速度；转向不能即时产生侧向加速度的物理限制仍然存在。

名义参考：
```math
e_theta=wrap(atan2(u_y,u_x)-theta_i),
b_ref=(min(||u_i||,s_max) max(cos(e_theta),0)-s_i)/tau,
omega_ref=k_theta e_theta.
```

约束包含纵向加速度、角速度、整个控制区间线速度，以及两端左右轮速：
```math
w_L/R(t)=(s_i+b_i t +/- L omega_i/2)/r_w.
```
轮速和线速度在恒定输入区间内仿射，端点约束覆盖整个区间。执行端不再限幅改变安全输出。

位置用恒定 b、omega 的解析积分更新：
```math
Delta p_x+i Delta p_y=exp(i theta_i)[s_i J0+b_i J1],
J0=integral_0^dt exp(i omega_i t) dt,
J1=integral_0^dt t exp(i omega_i t) dt.
```

对应：`dynamics.py:nominal_control, input_halfspaces, propagate`。

## 7. HOCBF 与局部责任分配

取 lambda1、lambda2>0：
```math
psi0=h, psi1=dot h+lambda1 h,
dot psi1+lambda2 psi1 >= margin.
```
连续时间保证还要求安全初始集合、输入可行和模型一致。常数 margin 是数值设计裕量，未被解释为任意扰动和采样延迟的严格鲁棒界。

机器人对：
```math
h_ij=||r_ij||^2-D^2,
c_ij=-2||v_i-v_j||^2-(lambda1+lambda2) dot h_ij-lambda1 lambda2 h_ij+margin,
R_i=2r_ij^T B_i, R_j=-2r_ij^T B_j,
R_i q_i+R_j q_j >= c_ij.
```

共同保守制动参考为 qbar_i=(-min(b_shared,s_i/dt),0)。令：
```math
epsilon=c_ij-R_i qbar_i-R_j qbar_j,
R_i q_i >= R_i qbar_i+alpha epsilon,
R_j q_j >= R_j qbar_j+(1-alpha) epsilon.
```
epsilon<=0 时 alpha=1/2；epsilon>0 时按双方保守输入盒相对于制动参考的改进能力分配。无正向能力的静止端不会承担不可能的正责任。两端相加仍等于联合约束，但不能由此推断每个局部问题永远可行。

静态凸矩形：
```math
h_o=dist(p_i,O)-d_o,
dot h_o=n_o^T v_i,
n_o^T B_i q_i >= -(lambda1+lambda2)n_o^T v_i-lambda1 lambda2 h_o+margin.
```
保守舍去外部距离的非负曲率，避免把平边切向速度错误当成离障加速度。边/角切换不是全局 C2；另用实际轨迹区间审计检验。

局部优化：
```math
min_q 0.5(q-q_ref)^T H_i(q-q_ref),
H_i=B_i^T B_i+diag(0.05,0.12),
subject to local barriers and actuator halfspaces.
```
二维 QP 枚举内部点、边驻点与顶点，并检查残差；集中式解相应 2N 维联合问题。失败时明确记录并制动，不伪装成成功。制动尚不是经过证明的通用恢复控制器。

对应：`safety.py`、`geometry.py`。

## 8. 区间审计、完成率和输出

精确轨迹按间隔 delta 采样，使用距离函数的 Lipschitz 性覆盖采样之间：
```math
d_pair(t)>=min_sample d_pair-(smax_i+smax_j)delta/2,
d_obstacle(t)>=min_sample d_obstacle-smax_i delta/2.
```
速度最大值由输入保持区间端点求得。不能认证物理间隔时保存失败片段并中止；不投影位置。

区域完成率与槽位完成率分开。槽位采用最终角色条件最优匹配和固定 0.35 m 阈值；运输成本为 (1/2N)sum_i||p_i-y_sigma*(i)||^2。图、GIF、CSV、JSON 和 NPZ 都由实际轨迹生成。

对应：`experiment.py:audit_hold, run`、`assignment.py:transport`、`visualization.py`。

## 9. 范围

本项目是本地信息接口上的单进程仿真，不是已部署的 ROS 网络。静态已知地图、理想实时安全感知、精确执行模型是当前验证条件。局部记忆、通信丢包实验不能被解释成动态障碍、传感器遮挡或异步安全的通用保证。

