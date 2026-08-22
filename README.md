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
- 有限 workload 的时间压缩/扩张和 capacity 搜索；
- sampled monotonicity verification；
- 当搜索上限仍然 safe 时，以 right-censored lower bound 报告 `C >= max_intensity`，而不是抛出异常或伪造 unsafe 点；
- 首次 violation、峰值并发、链路 headroom 和节点显存记录；
- 两条 Pipeline 的最小对照实验。

## 快速运行

项目核心无第三方运行依赖：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sla_aware_mvp.demo
```

当前示例比较：

1. `compute-first-slow-network`：更多层放在较快 GPU 上，但经过两条 0.5 Gbps 链路；
2. `network-aware-fast-link`：计算稍慢，但只经过一条 10 Gbps 链路。

当前固定示例的输出约为：

| Pipeline | Safe intensity 下界 | Safe arrival rate 下界 | 首个 unsafe 瓶颈 |
|---|---:|---:|---|
| compute-first-slow-network | 1.0859 | 约 1.61 req/s | Network (`a-b`) |
| network-aware-fast-link | 1.3750 | 约 2.04 req/s | SLA-time (`prefill`) |

数值来自 E 级 analytical/estimated profile，只用于验证 Evaluator 闭环，不能作为真实硬件性能结论。

## 当前建模假设

- 内部单位统一为秒、字节、字节/秒。
- Prefill duration 在 Arrival batch 后计算并冻结，后续 Arrival 不重放 Prefill。
- Decode progress 只按 compute time 推进；网络是独立的保守服务承诺，不模拟实际传输完成时间。
- Decode block 内使用 context 上界估计 compute 和 KV。
- Prefill 从 Arrival 开始预留完整 prompt KV。
- Pipeline compute latency 采用各 Stage 计算时间之和；并发干扰由透明的 analytical contention 系数近似。
- Queue overhead 不显式模拟，默认由 SLA 配置给出。
- Activation 大小按 `hidden_size * element_bytes * token_count` 推导。
- Workload intensity `lambda` 只压缩/扩张 inter-arrival 时间，不改变输入输出长度。
- 有限 workload 的 capacity 只在实际搜索区间内观测。若 `max_intensity` 仍 safe，则只声称 `C >= max_intensity`，不声称已找到真实 frontier。

这些假设集中在 `EvaluatorConfig` 和 `AnalyticalProfiler`，未来可以用 HELIX/实测 adapter 替换，不需要改事件引擎。

## 数据可信度

当前 demo 使用：

- Qwen2.5-7B 结构字段：来自 `raw_data/model.md`；
- GPU 算力：人为设置的敏感性实验值；
- Profile：analytical estimate；
- Workload：小型确定性合成轨迹。

因此当前整体标记为：

```text
E = estimated / analytical
```

后续数据层应保留：M（实测）、P（公开实测代理）、E（估计）、S（纯敏感性值）四类标签。

## 这个 MVP 能回答什么

它可以初步回答：

- SLA-derived network commitment 是否能识别 deployment-induced 网络瓶颈；
- context、并发和 KV 增长是否能在事件轨迹中触发动态红线；
- 不同 Pipeline 是否能得到不同且可解释的 capacity 排序；
- Evaluator 是否足够轻量，可以被后续 deployment search 多次调用。

它还不能证明：

- capacity 数值等于真实系统最大吞吐；
- 当前排序在真实 runtime 中必然成立；
- SLA-aware deployment 一定优于 HELIX；
- 这是一个足够强的论文贡献。

研究方向是否值得继续，下一阶段最关键的判据不是把模拟器写得更大，而是：使用 HELIX/Azure/BurstGPT 数据后，Evaluator 对 Pipeline 的排序是否与更真实 serving evaluation 有稳定相关性，并且 SLA-aware search 是否能找到更高 good-throughput 的最终部署。

## 下一步最小增量

1. 接入 HELIX LLaMA-2 70B 的 per-layer prompt/decode lookup，严格复现 ms 转秒和线性插值。
2. 接入一小段 Azure Conversation finite trace。
3. 构造 5-20 条固定 Pipeline，比较 capacity 排序。
4. 实现一个较细粒度 reference evaluator，检查两个排序的 rank correlation 与 first-violation 一致性。
5. 只有排序可靠后，再实现 deployment search。

## HELIX 公开数据实验

项目已固定 HELIX 官方 artifact commit：

```text
8639497a4aaf1eb3b7594614cb0bbd376c1342b3
```

仅 sparse-checkout 以下必要数据：

- LLaMA-2 70B + A100 40GB `prompt_bs2time.csv`；
- LLaMA-2 70B + A100 40GB `decode_bs2time.csv`；
- model-machine memory/inference metadata；
- Google machine profile；
- Azure Conversation interval counts和配对 input/output length 数据。

运行：

```bash
PYTHONPATH=src python3 -m sla_aware_mvp.helix_demo
```

固定 120 秒窗口生成 133 个有限请求，compute-only 默认语义下当前结果约为：

| Placement | Safe intensity 下界 | Safe arrival rate 下界 | 首个瓶颈 |
|---|---:|---:|---|
| 2.5 Gbps 慢链路顺序 | 0.4265 | 0.490 req/s | Network |
| 10 Gbps 快链路顺序 | 1.2279 | 1.411 req/s | Network |

两个候选使用相同的 8 个 A100、相同的 80 层切分和相同 workload，只改变 Stage 到物理节点的排列及由此经过的链路。快链路 placement 的保守 capacity 下界约为慢链路 placement 的 2.88 倍。

实验数据质量是混合的：

```text
M: HELIX LLaMA-2 70B / A100 per-layer profile
Generated: HELIX Azure interval counts + paired length distribution
S: 对照拓扑、链路容量和 SLA 设置
```

HELIX workload 文件不是原始 Azure request timestamp；当前 loader 使用连续的 3 秒 interval count、按目标均值缩放，并在区间内确定性放置请求。Pickle 通过禁止任何 class/global 构造的受限 unpickler 读取。

HELIX decode 表只观察 active decode-token batch，不观察 context length，也没有 `N_P/N_D` 混部维度。因此这个实验支持“deployment-induced network path 会显著改变 SLA-safe capacity”的研究信号，但还不能验证“context 增长使 TPOT 剩余预算下降”的完整动态耦合假设。

## Reference Evaluator v0

`feat/reference-evaluator-v0` 新增一个机械上独立的 reference execution model：显式 GPU Stage queue、Decode microbatch、physical-link FCFS `D/B` transfer，以及从执行轨迹测得的 TTFT/TPOT。它用于验证 Conservative Evaluator 是否保留固定 Pipeline 的排序，而不是声称复现真实 vLLM/HELIX scheduler。

第一次 30 秒 smoke attempt 发现：只有 17 个请求时，Conservative search 在 `max_intensity=16` 仍然 safe。该情况现在被正式建模为 right-censored capacity observation，smoke harness 会继续运行 Reference，而不是因“找不到 unsafe”提前退出。
