# Conservative SLA-Safe Evaluator MVP

这是固定 Layer-Level LLM Pipeline Evaluator 的最小研究原型。它不是高保真 serving simulator，目标是先验证以下核心机制是否能对候选部署产生有解释力的差异：

```text
request state
-> profiling-based compute time
-> remaining TTFT/TPOT budget
-> minimum physical-link bandwidth commitment
-> network / memory / SLA-time red lines
-> maximum safe finite-workload intensity
```

## MVP 包含什么

- 固定、连续 Layer Stage 的 Pipeline 数据模型与路径校验；
- Arrival、Prefill-to-Decode、Decode Block Update、Finish 事件；
- Atomic Prefill；
- 连续 equivalent-token Decode progress；
- Decode block 上界 context，用于 compute 和 KV 的保守更新；
- SLA 剩余时间反推逐链路最低带宽，并对 active requests 求和；
- 权重、workspace、安全余量和动态 KV 显存红线；
- 有限 workload 的时间压缩/扩张；
- sampled monotonicity verification + local refinement 的 capacity search；
- 同时 violation、峰值并发、链路 headroom 和节点显存记录；
- HELIX 公开 profile + Azure-derived finite workload 的最小对照实验。

## 快速运行

项目核心无第三方运行依赖：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sla_aware_mvp.demo
```

## 当前默认状态推进语义

默认使用：

```text
progress_policy = compute_only
```

含义是：

- Prefill/Decode 的状态推进由 profiling-based compute/service time 驱动；
- 网络仍按剩余 TTFT/TPOT 时间反推最低 bandwidth commitment；
- 网络 commitment 是保守 red-line 检查，不直接减慢状态机；
- `EvaluatorConfig.conservative_network_lifetime=True` 仅保留为 `full-SLA-window lifetime ablation`，不是默认模型。

一次受控消融显示，Full-SLA-window policy 会显著改变并发轨迹和 capacity，因此当前不把它作为 P0 正确修复。真正应该验证的是：哪种 coarse timing abstraction 能更好保持不同 Pipeline 相对于独立 reference evaluator 的排序。

## 当前建模假设

- 内部单位统一为秒、字节、字节/秒。
- Prefill duration 在 Arrival batch 后计算并冻结，后续 Arrival 不重放 Prefill。
- Decode progress 默认按 profiling compute/service time 推进。
- Decode block 内使用 context 上界估计 compute 和 KV。
- Prefill 从 Arrival 开始预留完整 prompt KV。
- Pipeline compute latency 采用各 Stage 计算时间之和；并发干扰由 profiler 提供。
- Queue overhead 不显式模拟，默认由 SLA 配置给出。
- Activation 大小按 `hidden_size * activation_element_bytes * token_count` 推导。
- Workload intensity `lambda` 只压缩/扩张 inter-arrival 时间，不改变输入输出长度。

## Capacity search

`find_capacity()` 不再直接假设 feasibility 全局单调并纯二分，而是：

1. 找到一个 sampled safe/unsafe bracket；
2. 在 bracket 内做 coarse grid；
3. 在更高 intensity 上继续检查明显的 safe re-entry；
4. 如果 sampled points 出现 `unsafe -> safe`，停止并报错；
5. 仅在 sampled monotonic 后做局部 refinement。

返回值中的 `monotonicity_verified_on_samples=True` 只表示采样点上未发现反例，不是数学证明。

## 数据可信度

当前 analytical demo 使用估计 profile，只用于验证闭环；HELIX demo 使用：

```text
M: HELIX LLaMA-2 70B / A100 per-layer profile
Generated: HELIX Azure interval counts + paired length distribution
S: 对照拓扑、链路容量和 SLA 设置
```

HELIX workload 文件不是原始 Azure request timestamp；当前 loader 使用连续的 3 秒 interval count、按目标均值缩放，并在区间内确定性放置请求。

## HELIX 公开数据实验

项目固定 HELIX 官方 artifact commit：

```text
8639497a4aaf1eb3b7594614cb0bbd376c1342b3
```

运行：

```bash
PYTHONPATH=src python3 -m sla_aware_mvp.helix_demo
```

当前默认 HELIX demo 使用 compute-only progress。最近一次受控实验得到：

| Placement | Safe intensity 下界 | Safe arrival rate 下界 | 首个 unsafe 瓶颈 |
|---|---:|---:|---|
| 2.5 Gbps 慢链路顺序 | 0.4265 | 0.490 req/s | Network |
| 10 Gbps 快链路顺序 | 1.2279 | 1.411 req/s | Network |

两个 placement 的首次 unsafe event 都是七条同容量 stage link 同时越线，因此兼容字段 `bottleneck_object=slow-0/fast-0` 不能解释成唯一瓶颈，应查看完整 `first_violations`。

Full-SLA-window lifetime ablation 得到约：

```text
Slow: 0.338 req/s
Fast: 1.309 req/s
```

这组结果只用于说明 lifetime abstraction 会显著影响状态轨迹，不作为当前默认 capacity 结论。

HELIX decode 表只观察 active decode-token batch，不观察 context length，也没有混合 Prefill/Decode 干扰维度。因此当前实验仍不能验证 context 增长、混部干扰与真实 runtime ordering。

## 这个 MVP 能回答什么

当前可以初步回答：

- SLA-derived network commitment 是否能识别 deployment-induced 网络瓶颈；
- context、并发和 KV 增长能否进入事件轨迹并触发动态红线；
- 不同 Pipeline 是否能得到不同且可解释的 evaluator-defined capacity；
- Evaluator 是否足够轻量，可被后续 deployment search 多次调用。

当前还不能证明：

- capacity 等于真实系统最大吞吐；
- compute-only 是正确的真实 timing abstraction；
- 当前 Pipeline 排序在真实 runtime 中稳定；
- SLA-aware deployment 一定优于 HELIX；
- 当前 evaluator 已经可以直接作为 deployment oracle。

## 下一步决定性实验

下一阶段不继续膨胀 Conservative Evaluator，而是实现一个独立 reference serving evaluator，并在一组不同 Layer-Level Pipelines 上比较：

```text
C_compute-only
C_full-SLA-window
C_reference
```

重点比较：

- Spearman rank correlation；
- Kendall tau；
- Top-k overlap；
- first-violation category/set agreement；
- evaluation time / speedup。

只有当 Conservative Evaluator 对 reference 的 Pipeline 排序稳定后，再进入 deployment search。
