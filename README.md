# TAC_Control_v6.0

V5 的并列独立升级项目。实现可执行差速输入安全过滤、本地观测与记忆、局部路径搜索和邻居让行；V5 源码与环境未修改。

## 首次运行

二维差速无人车仿真，Python 3.12；本版本不依赖 acados、ROS 或实体机器人。
发布副本不包含历史 output 和虚拟环境，以下结果文件在运行后生成。

```bash
# 安装 uv 后，在仓库根目录运行
uv sync --locked
uv run pytest -q
uv run amr-v6 --mode hybrid --robots 6 --horizon 1 --output output/quick --no-gif
# 完整双模式实验，再做独立审计
uv run amr-v6 --mode both --horizon 60 --output output/validation --no-gif
uv run python scripts/audit_outputs.py output/validation
```

uv 安装方式见 https://docs.astral.sh/uv/getting-started/installation/ 。
1 秒实验用于检查流程，不代表任务完成。原审计脚本要求双模式、全部到位和零降级，仅用于上述完整默认验证，不用于冒烟实验。GIF 为回放，不代表实机实时运行。

## 已验证结果

默认场景：24 台机器人，60 s，种子 20260803，控制周期 0.05 s。hybrid 每 3 步更新全局目标，协同消息丢包率 10%、延迟 1 步；安全观测独立且实时。

| 指标 | 集中式 | Hybrid |
|---|---:|---:|
| 区域完成率 | 100% | 100% |
| 槽位完成率（0.35 m 阈值） | 100% | 100% |
| 全部到位并持续保持的时刻 | 49.20 s | 48.15 s |
| 最终最大槽位误差 | 0.1581 m | 0.1578 m |
| 机器人间距的区间下界 | 0.6225 m | 0.6050 m |
| 障碍物净空的区间下界 | 0.3458 m | 0.3683 m |
| 安全降级次数 | 0 | 0 |
| 模拟有向消息发送数 | 720000 | 368894 |

物理碰撞阈值分别为 0.36 m 和 0.18 m；请求的 CBF 裕量分别为 0.56 m 和 0.30 m。上述区间下界同时满足两类阈值。消息发送数约减少 48.8%，不是网络字节量或真实网络实测。

这组结果针对明确的仿真条件，不代表所有地图、初始状态和传感误差条件下的通用保证。

## 查看动图和结果

- [warehouse.gif](output/validation/warehouse.gif)：双模式动图，包含机器人编号、实时槽位完成数和安全间隔。
- [index.html](output/validation/index.html)：双击打开离线交互回放，支持拖动时间、显示协同连线和目标匹配。
- [trajectories.png](output/validation/trajectories.png)：完整轨迹。
- [safety_diagnostics.png](output/validation/safety_diagnostics.png)：区间安全下界、任务误差与降级次数。
- [assignment_history.png](output/validation/assignment_history.png)：目标标签变化。
- [AUDIT.json](output/validation/AUDIT.json)：独立轨迹和输入审计。
- [summary.csv](output/validation/summary.csv)：主要指标；NPZ、逐步 CSV、JSON 同目录保存。
- [REPORT.md](output/validation/REPORT.md)：自动生成的结果说明。

旧试验与预览文件已清理，最终结果统一保存在 output/validation。

## 使用 uv 运行

```powershell
cd <仓库目录>
uv sync --locked
uv run amr-v6 --mode both
```

默认生成 PNG、GIF、HTML、CSV、JSON 和 NPZ，不需要另外打开动图开关。

```powershell
# 只对已保存的轨迹重新生成全部图和动图
uv run amr-v6 --mode both --render-only

# 自动测试
uv run pytest -q

# 独立审计最终输出
uv run python scripts/audit_outputs.py output/validation

# 短时运行，不生成动图
uv run amr-v6 --mode hybrid --robots 12 --horizon 5 --output output/quick --no-gif
```

Python 3.12 环境由 uv 新建在 .venv，uv.lock 固定依赖。Windows 中文路径采用 Hatch 精确可编辑导入与 editables 依赖；不依赖 V5 环境。uv build 可生成独立 wheel，地图资源随包提供。

## 本次解决的关键问题

1. 安全层直接优化纵向加速度和角速度，将加速度、速度、转向与轮速限制放在优化内部；实际执行不再限幅改变安全结果。
2. 采用递归 HOCBF 的正参数和/积；矩形障碍使用保守支持法向，去掉错误的切向速度增益。
3. 本地二维 QP 求精确候选最优解并检查残差；求解失败显式记录并制动，无法认证物理间隔则停止并保存失败片段。
4. 互惠责任围绕共同制动参考、按可用输入能力分配，避免要求静止机器人执行不可能的避让。
5. LocalView 将实际安全观测和延迟协同消息分开；拥挤项使用有限支撑势函数，只依赖局部观测。
6. 每台机器人独立搜索局部观测路径，并保留停车邻居的位置记忆，避免一离开视野就重新选择被堵通道。
7. 邻居之间交换 blocked/yielding 状态，已到位机器人可暂时让出通道。临时路径点不修改官方目标。
8. 初始化修复 V5 跨角色目标仅相距约 4.9 cm 的物理重叠；角色及目标区域保留，槽位在运行期间固定。
9. 精确恒定输入轨迹配合 Lipschitz 区间下界审计；没有通过位置投影、直接移到目标或放宽 0.35 m 完成阈值制造结果。

## 代码结构

```text
src/amr_control/
  config.py           统一配置
  scenario.py         群体能力、角色与相容目标槽位
  geometry.py         矩形距离、支持法向与线段距离
  navigation.py       静态地图和角色导航场
  local_planner.py    每个机器人的路径、观测记忆与临时让行
  assignment.py       低频条件分配和运输指标
  communication.py    范围传感、延迟队列、消息缓存与 LocalView
  guidance.py         Graphon、有限支撑拥挤和期望速度
  dynamics.py         可执行输入、精确积分、轮速
  safety.py           集中/本地 HOCBF 和失败处理
  experiment.py       世界适配、闭环调度与区间审计
  visualization.py    图、GIF、HTML 和报告
  cli.py              运行入口
tests/                33 项自动测试
scripts/audit_outputs.py  最终输出独立审计
docs/                 数学模型、迁移说明与边界
```

完整模型见 [MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md)，修改说明见 [MIGRATION.md](docs/MIGRATION.md)。

## 验证边界

这是本地信息接口上的单进程仿真，不是已部署的 ROS 2 多机系统。静态地图已知；安全传感器无噪声、无遮挡、同步。协同通信有丢包和延迟；导航记忆中的预测状态不进入 CBF。

局部路径暂时找不到通路时会等待、重规划或协商让行；local_route_failures 记录这些规划事件，不等于安全优化降级。

未宣称全局分布一致性估计、任意场景无死锁、动态障碍鲁棒安全或完整闭环能量单调。现有局部位置记忆与 KDE 不被描述成完整全局概率分布估计。细节见 [VALIDATION.md](docs/VALIDATION.md)。


## 2026-09-15 计算与网络可视化优化

局部规划增加候选代价剪枝：已不可能优于当前可行候选的连接不再重复检查几何可见性。40 个保存状态、3 次交替顺序测试中，规划中位耗时由 1.052 s 降至 0.443 s，约加快 2.37 倍；这是规划模块基准，不是整套系统或机器人到达速度提升。

两种模式完整重跑后，所有保存数组与优化前完全一致。集中式 / hybrid 全部持续到位时间仍为 49.20 / 48.15 s，零安全降级。路径长度和停等指标见 [对比数据](output/optimized/COMPARISON.json)，基准见 [PLANNER_BENCHMARK.json](output/optimized/PLANNER_BENCHMARK.json)。可通过 `uv run python scripts/benchmark_planner.py` 复测。

最终动图仍是 [warehouse.gif](output/validation/warehouse.gif)：上排为实际选用的协同作用网络，箭头从消息发送者指向接收者；下排为理想范围感知关系及当前目标匹配。感知边由保存位置和传感半径重建，不等于全部活跃 CBF 约束。GIF 抽取 240 帧；交互回放保存全部 1201 个状态，可按 0.05 s 单步查看。终点网络显示最后一次控制步的协同关系。

[交互回放](output/validation/index.html)默认显示协同箭头，可切换感知和目标关系。协同边表示被控制器选用的有效消息，不表示该时刻所有发送、丢失和在途数据包；目前未保存逐包事件历史。

## 三行两列网络动图

最新版 warehouse.gif 为 3×2 布局：左列集中式、右列 hybrid；第一排为同角色 Graphon 协同网络，第二排为安全感知与目标匹配，第三排为跨角色三个层次的作用关系。第三排绿色为拥挤项，紫色点线为导航使用的观测/记忆邻居，橙色为实际触发且仍处于让行期间的请求关系；淡红虚线为纳入的机器人对 CBF 约束，粗红线为约束余量绝对值不超过 1e-5 的紧约束。

第三排按“任意一个方向存在即画一条线”显示机器人对；NPZ 保存完整有向关系。规划输入不等于已证明引发绕行，紧约束也不等于具有正 KKT 乘子的因果约束。终点沿用最后一个控制步的网络。新增 interaction.py 只记录关系，不参与控制决策；测试覆盖同角色排除、跨角色方向、导航记忆和实时安全感知的区别。

通过 `uv run python scripts/compare_modes.py` 生成 [模式对比图](output/validation/mode_comparison.png) 和 [完整对比数据](output/validation/MODE_COMPARISON.json)。混合式优势以同一场景的消息发送数、任务表现和局部问题规模为依据，不把单进程运行时间解释成真实多机并行性能。全局分配仍依赖中心节点。
