# 固定流水线 Conservative SLA-Safe Evaluator：HELIX 数据 MVP 实验报告

## 0. 文档目的

本文档描述一个用于验证研究方向可行性的最小实验。研究目标不是立即证明新方法优于现有系统，而是先回答：

> 对固定的 Layer-Level LLM Pipeline，能否通过一个低成本、事件驱动、SLA-aware 的 Conservative Evaluator，对不同物理放置产生有区分度、可解释的 SLA-safe capacity 排序？

本文档完整说明实验问题、数据来源、建模假设、实验流程、结果和局限，可直接交给其他 ChatGPT 对话或研究人员进行独立分析。

---

## 1. 研究背景与 Evaluator 定位

最终研究目标是在异构 GPU 和受限物理网络中寻找 Layer-Level LLM deployment，使真实动态 workload 下满足 TTFT/TPOT SLA 的 good throughput 更高。

当前实现的 Evaluator 不是高保真 serving simulator，也不声称返回真实系统的绝对最大吞吐。它的定位是部署搜索阶段的低成本评价函数：

```text
Fixed Pipeline
    + Finite Workload
    + TTFT / TPOT SLA
    + Compute / Network / Memory Profile
                    ↓
Event-Driven Conservative Evaluator
                    ↓
Conservative SLA-Safe Workload Capacity
```

Evaluator 的核心逻辑是：

```text
当前请求状态
→ Profiling-based Compute Time
→ TTFT/TPOT 剩余时间预算
→ 反推最低物理链路服务能力
→ 聚合所有 active requests 的带宽承诺
→ 检查 Network / Memory / SLA-time 红线
```

Evaluator 返回的 capacity 是有限 workload generation protocol 下的安全负载强度，不是无限时间随机排队系统的稳定吞吐。

---

## 2. 本次 MVP 要验证的假设

本次实验只验证最基础的研究信号。

### H1：物理网络路径会改变 SLA-safe capacity

在模型、GPU 数量、Layer 切分、workload 和 SLA 完全相同时，仅改变 Stage 到物理节点的排列及相邻 Stage 经过的链路，Evaluator 应当给出不同 capacity。

### H2：Evaluator 能解释首次失败原因

当 workload intensity 超过安全边界时，Evaluator 应记录首次 violation 的时间、类型、物理对象、需求、容量和系统并发状态，而不只是返回一个布尔值。

### H3：更高带宽的 placement 应具有更高 capacity

在两个候选方案 compute 和 memory 条件相同的情况下，经过 10 Gbps 链路的方案应比经过 2.5 Gbps 链路的方案支持更高的 SLA-safe request rate。

这三个假设成立只能证明 Evaluator 的最小闭环有区分能力，不能证明它的排序与真实 serving runtime 一致。

---

## 3. 实验数据

### 3.1 HELIX 官方 Artifact

数据来自 HELIX 官方仓库：

- Repository: https://github.com/Thesys-lab/Helix-ASPLOS25
- 固定 commit: `8639497a4aaf1eb3b7594614cb0bbd376c1342b3`
- License: Apache-2.0

本项目没有下载和运行完整 HELIX simulator，只 sparse-checkout 了实验所需的最小文件集合：

```text
simulator/model_manager/llama2_70b/a100/prompt_bs2time.csv
simulator/model_manager/llama2_70b/a100/decode_bs2time.csv
simulator/model_manager/llama2_70b/a100/llama2_70b_a100.py
simulator/model_manager/google_machine_profiles.ini
simulator/event_simulator/utils.py
simulator/trace_generator/arrival_rate/azure_conv_arrive_time.pkl
simulator/trace_generator/length_data/azure_conv_input.pkl
simulator/trace_generator/length_data/azure_conv_output.pkl
```

原始文件保存在：

```text
data/raw/helix/Helix-ASPLOS25/
```

commit、文件校验和和数据角色记录在：

```text
data/provenance/sources.yaml
```

### 3.2 Compute Profile

使用 HELIX 的：

```text
LLaMA-2 70B + A100 40GB
```

profile 数据语义为单个模型 layer 的执行时间。CSV 第二列单位是毫秒，接入时乘以 `0.001` 转换为秒，并严格复现 HELIX 的分段线性插值。

Prompt profile：

```text
x = prompt token profiling point
y = seconds per layer
有效范围 = [0, 11250]
```

代表性数据：

| Prompt tokens | Time per layer |
|---:|---:|
| 250 | 0.002 s |
| 750 | 0.005 s |
| 1000 | 0.008 s |
| 2000 | 0.014 s |

Decode profile：

```text
x = active decode-token batch point
y = seconds per layer
有效范围 = [0, 800]
```

代表性数据：

| Active decode tokens | Time per layer |
|---:|---:|
| 1（在 0 和 2 之间插值） | 0.0007 s |
| 10 | 0.0014 s |
| 20 | 0.0015 s |
| 50 | 0.0017 s |
| 100 | 0.0025 s |

重要限制：HELIX decode 横轴不是 context length。该 profile 没有观测以下维度：

```text
context_length
N_prefill / N_decode mixed interference
scheduler contention
```

因此当前 HELIX provider 不会伪造这些维度，也不会把 HELIX 数据重新标记为其他模型或 GPU 的实测数据。

### 3.3 HELIX Azure Conversation Workload

HELIX artifact 提供：

- 1200 个连续的 3 秒 arrival-count interval；
- 16657 对 Azure Conversation input/output token length；
- HELIX 预处理后的长度约束。

Pickle 文件没有直接使用普通 `pickle.load`。项目实现了受限 Unpickler，禁止所有 class/global 构造，只接受 primitive list 和整数元素。

本次 workload 生成参数：

```text
arrival source       = Azure Conversation interval counts
length source        = Azure Conversation paired input/output lengths
window duration      = 120 s
target mean rate     = 1.5 req/s（按完整 1200 interval 均值缩放）
random seed          = 7
interval offset      = 0
```

生成方法：

1. 顺序读取前 40 个 3 秒 interval，保留局部时间形状；
2. 根据完整 1200 interval 的均值缩放每个 interval 的 request count；
3. 使用 residual rounding 避免逐 interval 取整造成长期系统性偏差；
4. 在每个 3 秒 interval 内等间隔放置请求；
5. 使用固定 seed，从配对 input/output length 数组中采样。

最终有限 workload：

| 指标 | 数值 |
|---|---:|
| 请求数 | 133 |
| 第一个 Arrival | 4.5 s |
| 最后一个 Arrival | 119.4 s |
| 实际基础 arrival rate | 1.148825 req/s |
| Input tokens 最小值 | 28 |
| Input tokens 中位数 | 1009 |
| Input tokens 平均值 | 823.3233 |
| Input tokens 最大值 | 2000 |
| Output tokens 最小值 | 12 |
| Output tokens 中位数 | 150 |
| Output tokens 平均值 | 222.8797 |
| Output tokens 最大值 | 626 |

实际窗口请求率低于目标 1.5 req/s，是因为选中的前 40 个 interval 低于完整 1200 interval 的全局均值。这一局部波动被保留，而没有二次强制校正。

这不是原始 Azure request timestamp trace，而是从 HELIX interval counts 和长度分布生成的确定性有限轨迹。

### 3.4 数据可信度标签

本实验包含不同可信度的数据：

| 内容 | 标签 | 含义 |
|---|---|---|
| HELIX LLaMA-2 70B/A100 profile | M | 公开 artifact 中匹配模型/GPU的实测 profile |
| Azure-derived workload | Generated | 基于 HELIX 数据生成，不是原始时间戳 |
| 物理拓扑和链路容量 | S | 为对照实验设置的 sensitivity topology |
| TTFT/TPOT SLA | S | 人为设置的研究参数 |
| KV memory 公式 | E | 基于模型结构的 analytical estimate |

因此不能把整个实验简称为“真实 HELIX 实验”，准确表述应为：

> 使用 HELIX 公开 compute profile 和 Azure-derived workload，在合成对照拓扑与 SLA 下进行的 Evaluator MVP 实验。

---

## 4. 模型、基础设施和候选 Pipeline

### 4.1 模型

```text
Model                    = LLaMA-2 70B
Layers                   = 80
Parameters               = 70B
Weight precision         = 2 bytes/parameter
Hidden size              = 8192
Attention heads          = 64
KV heads                 = 8
KV element size          = 2 bytes
```

### 4.2 GPU 节点

```text
GPU nodes                = 8
GPU type                 = A100 40GB
Layers per node          = 10
Total layers             = 8 × 10 = 80
```

HELIX A100 metadata给出的 `max_num_layers = 12`，因此每节点放置 10 层没有超过该限制。

每个候选方案都使用同样的 8 个节点、相同的 10-layer Stage 切分，Compute Profile 和固定权重占用完全相同。

### 4.3 Slow-Link Placement

Stage 对节点的顺序：

```text
n0 → n2 → n4 → n6 → n1 → n3 → n5 → n7
```

相邻 Stage 之间经过 7 条有向链路：

```text
slow-0 ... slow-6
capacity = 2.5 Gbps = 312,500,000 bytes/s per link
```

### 4.4 Fast-Link Placement

Stage 对节点的顺序：

```text
n0 → n1 → n2 → n3 → n4 → n5 → n6 → n7
```

相邻 Stage 之间经过 7 条有向链路：

```text
fast-0 ... fast-6
capacity = 10 Gbps = 1,250,000,000 bytes/s per link
```

### 4.5 控制变量

两个候选方案保持一致的变量：

- 模型和模型精度；
- GPU 类型和 GPU 数量；
- 每个 Stage 的 layer 数；
- 所有 Stage 的 compute time；
- 固定模型权重和 KV 计算方式；
- workload；
- TTFT/TPOT SLA；
- Decode block size；
- Evaluator 算法。

唯一主要差异是 Stage 物理排列导致的链路容量差异。

---

## 5. SLA 和 Evaluator 配置

```text
TTFT SLA                 = 2.000 s
TPOT SLA                 = 0.150 s/token
Fixed overhead           = 0.005 s
Queue overhead           = 0
Decode block size        = 16 tokens
Resource context policy  = block upper bound
```

### 5.1 请求状态推进

事件类型：

```text
Arrival
Prefill → Decode
Decode Block Update
Finish
```

Prefill 使用 Atomic Prefill：Arrival batch 后计算 Prefill duration 并冻结，后续 Arrival 不重放已启动的 Prefill。

Decode 使用连续 equivalent-token progress：

```text
g_r(t_next) = g_r(t) + (t_next - t) / T_decode_per_token
```

资源状态只在 16-token block boundary 或其他系统事件发生后更新。Block 内 compute context 和 KV 使用该 block 的 context 上界。

同一 timestamp 上的多个事件作为一个原子 batch 处理，不检查程序执行顺序产生的人工中间状态。

### 5.2 SLA 剩余网络时间

对 active request `r`：

```text
Delta_r = SLA_r - T_compute,r - T_queue,r - T_fixed,r
```

- Prefill 使用 TTFT SLA；
- Decode 使用 TPOT SLA；
- 若 `Delta_r <= 0`，立即产生 SLA-time violation。

### 5.3 网络最低服务需求

请求在物理链路 `e` 上传输的数据量为 `D_e,r`，链路容量为 `B_e`。

按链路归一化传输成本分配剩余网络时间：

```text
w_e,r = (D_e,r / B_e) / sum_j(D_j,r / B_j)
delta_e,r = w_e,r × Delta_r
b_required_e,r = D_e,r / delta_e,r
```

物理链路上的保守总承诺：

```text
B_required_e(t) = sum_{r in Active(t)} b_required_e,r(t)
```

红线：

```text
B_required_e(t) <= B_e
```

Evaluator 不利用精确通信错峰、统计复用或 activation 传完后的瞬时释放。

### 5.4 Memory 红线

节点 `i`：

```text
M_used_i(t)
= M_weight_i
+ M_workspace_i
+ M_margin_i
+ sum_active_requests M_KV_i,r(t)
```

红线：

```text
M_used_i(t) <= M_capacity_i
```

本次实验中显存不是 capacity 边界的主要瓶颈。

---

## 6. Workload Intensity 与 Capacity 搜索

设基础有限 workload 为 `W_0`，强度参数为 `lambda`。

生成 `W_lambda` 时只缩放相对 Arrival 时间：

```text
t_arr(lambda)
= t_origin + (t_arr(1) - t_origin) / lambda
```

Input/output length、请求顺序和请求数量不变。

对于每个 `lambda`：

1. 从第一个 Arrival 开始运行；
2. Arrival 停止后继续推进；
3. 直到所有 active requests Finish；
4. 任意状态发生 SLA-time、Network 或 Memory violation，则该 `lambda` unsafe；
5. 完整 Drain 且从未违反红线，则该 `lambda` safe。

Capacity 搜索先指数扩展 unsafe 上界，再在 safe/unsafe 区间内二分。当前相对容差为 `0.01`。

最终报告：

```text
safe_intensity_lower_bound
unsafe_intensity_upper_bound
```

对应请求率通过以下方式换算：

```text
safe_request_rate_lower_bound
= base_window_rate × safe_intensity_lower_bound
```

---

## 7. 实验结果

### 7.1 Capacity 结果

| Placement | Safe intensity 下界 | Unsafe intensity 上界 | Safe request rate 下界 | Capacity trials | 首个瓶颈 |
|---|---:|---:|---:|---:|---|
| Slow-Link | 0.4453125 | 0.4531250 | 0.511586 req/s | 7 | Network `slow-0` |
| Fast-Link | 1.2734375 | 1.2812500 | 1.462957 req/s | 11 | Network `fast-0` |

Fast-Link 相对于 Slow-Link 的安全请求率下界比例：

```text
1.462957 / 0.511586 ≈ 2.86×
```

### 7.2 最大安全状态附近

Slow-Link safe run：

```text
Peak Prefill concurrency       = 2
Peak Decode concurrency        = 22
Minimum slow-link headroom     = 8,651,272.73 bytes/s
Slow-link capacity             = 312,500,000 bytes/s
Peak link commitment ratio     ≈ 97.23%
Peak per-node memory           = 18,613,538,560 bytes
Memory utilization             ≈ 46.53% of 40GB
```

Fast-Link safe run：

```text
Peak Prefill concurrency       = 4
Peak Decode concurrency        = 52
Minimum fast-link headroom     = 95,203,458.55 bytes/s
Fast-link capacity             = 1,250,000,000 bytes/s
Peak link commitment ratio     ≈ 92.38%
Peak per-node memory           = 19,961,368,320 bytes
Memory utilization             ≈ 49.90% of 40GB
```

Fast-Link 方案支持了更高的并发状态，因此其安全边界附近显存占用也更高，但仍未达到显存红线。

### 7.3 首个 Unsafe Violation

Slow-Link：

```text
lambda                        = 0.453125
violation time                = 155.672414 s
violation type                = Network
physical link                 = slow-0
required bandwidth            = 313,555,862.07 bytes/s
link capacity                 = 312,500,000.00 bytes/s
required / capacity           ≈ 100.34%
N_prefill                     = 1
N_decode                      = 13
```

Fast-Link：

```text
lambda                        = 1.28125
violation time                = 91.779586 s
violation type                = Network
physical link                 = fast-0
required bandwidth            = 1,447,253,333.33 bytes/s
link capacity                 = 1,250,000,000.00 bytes/s
required / capacity           ≈ 115.78%
N_prefill                     = 0
N_decode                      = 53
```

Fast-Link 的 unsafe 点越过红线幅度较大，是因为资源只在离散事件状态检查，邻近两个 intensity 可能在某个事件上产生不同的并发组合，而不是连续平滑地贴近容量。

---

## 8. 初步结果解释

### 8.1 当前实验支持的结论

1. **H1 得到支持**：模型、GPU、Layer 切分、workload 和 SLA 相同，仅改变物理网络路径，capacity 出现明显差异。
2. **H2 得到支持**：Evaluator 不仅返回 safe/unsafe，还定位了首次失败链路、需求、容量和并发状态。
3. **H3 得到支持**：10 Gbps placement 的安全请求率下界约为 2.5 Gbps placement 的 2.86 倍。
4. 网络友好 placement 可以进入更高 Prefill/Decode 并发区间，显存占用随之上升，但本实验中显存仍不是主要瓶颈。
5. 使用 HELIX 公开 per-layer profile 后，Evaluator 的事件、profile、SLA budget 和网络承诺闭环可以稳定运行，不依赖人为直接填写请求计算时间。

### 8.2 当前实验最有价值的研究信号

结果表明，仅依赖计算性能或静态 GPU 容量进行 Layer-Level placement，可能无法反映 SLA 下真实的网络服务压力。把 compute time 放入 TTFT/TPOT budget 后，剩余网络时间会随请求状态变化，并使物理路径差异转化为可比较的 capacity score。

但由于本实验特意让两个候选的主要差异集中在链路容量，得到“快链路 capacity 更高”仍然具有一定的构造性。它验证了实现逻辑和可解释性，但尚未验证 Evaluator 排序的外部有效性。

---

## 9. 当前实验不能证明什么

### 9.1 不能证明 capacity 等于真实最大吞吐

Evaluator 的网络模型是保守带宽承诺求和，没有显式模拟：

- activation 的真实开始/结束时间；
- GPU 与网络 overlap；
- 微观通信错峰；
- batching 和 scheduler；
- 链路排队与动态带宽分配。

因此 capacity 是 Conservative score，不是 runtime throughput。

### 9.2 不能证明 Pipeline 排序与真实 serving 一致

当前只有两个候选，且网络差异明显。还没有与独立 reference simulator 或真实 runtime 比较 rank correlation。

### 9.3 不能验证 Decode context growth 的完整假设

HELIX decode profile 不包含 context length。当前代码虽然维护 Decode block context 和动态 KV，但 HELIX provider 的 decode compute time 不随 context 增长。

因此尚未验证：

```text
context 增长
→ decode compute time 增长
→ TPOT 剩余网络预算下降
→ required bandwidth 增长
→ 后期触发 violation
```

### 9.4 Decode progress 没有加入实际网络时间

当前 Decode progress 只使用 compute time 推进，网络只作为并行的资源承诺红线。如果通信不能充分 overlap，这可能低估请求 active lifetime 和并发数量，使结果偏乐观。

### 9.5 Workload 不是原始 Azure 时间戳

当前 workload 是从 HELIX interval counts 和长度分布生成的。它保留局部 burst shape，但不是原始逐请求 trace。

### 9.6 Capacity 依赖有限轨迹

请求数量固定为 133，增大 `lambda` 只压缩 Arrival 时间。结果不能解释为无限时间 stationary arrival process 的稳定容量。

### 9.7 SLA 和拓扑是人为设置

TTFT=2 s、TPOT=150 ms、2.5/10 Gbps 链路用于 MVP sensitivity experiment，尚未证明代表特定真实 edge deployment。

---

## 10. 判断研究方向是否有意义的下一步实验

### 10.1 最关键：与独立 Reference Evaluator 比较排序

生成 20-100 条不同 Layer placement，分别用：

1. 当前 Conservative Evaluator；
2. 更细粒度、显式模拟 GPU/network queue 和传输结束时间的 Reference Evaluator；

得到每条 Pipeline 的 capacity 或 good throughput 排名，并计算：

```text
Spearman rank correlation
Kendall tau
Top-k overlap
First-violation agreement
Evaluation wall-clock cost
```

如果 Conservative Evaluator 明显更快，并能稳定保持高排名相关性，才证明它有资格作为 deployment search oracle。

### 10.2 SLA-aware 机制消融

比较以下评价/搜索标准：

```text
A. Static compute + memory capacity
B. Static network bandwidth capacity
C. Compute-aware but SLA-unaware
D. 完整 SLA-derived network commitment
```

最终把各标准选出的 deployment 放到同一个 reference serving evaluation 中比较 good throughput。

### 10.3 Decode block size 敏感性

测试：

```text
q = 1, 8, 16, 32, 64
```

比较 capacity、排序和运行时间，判断 `q=16` 是否能在精度和成本之间取得合理折中。

### 10.4 SLA 与网络敏感性

进行二维 sweep：

```text
TTFT ∈ {1, 2, 4, 8} s
TPOT ∈ {50, 100, 150, 250} ms
Bandwidth ∈ {0.1, 0.5, 1, 2.5, 10, 25} Gbps
```

观察瓶颈是否在 compute、network 和 memory 之间合理迁移。

### 10.5 Context-aware Profile

加入明确标注为 E/P/M 的 context correction 或 context-aware profile，验证后期 Decode block 是否会触发新的 SLA/network violation。

### 10.6 Workload 稳健性

至少比较：

- HELIX Azure Conversation；
- HELIX Azure Code；
- BurstGPT 小窗口；
- 平滑 Poisson workload；
- bursty synthetic workload。

---

## 11. 建议分析者重点审查的问题

请重点分析以下问题：

1. `SLA remaining time → minimum link service requirement` 的数学逻辑是否成立？
2. 使用 `(D_e/B_e)` 归一化分配网络时间预算是否有合理理论依据，还是需要改为优化问题或 end-to-end service curve？
3. 对所有 active requests 的最低带宽承诺直接求和是否过度保守？这种保守性是否仍可能保持 Pipeline 排序？
4. Decode progress 只按 compute time 推进、网络不进入生命周期，会不会与“Conservative”定位矛盾？
5. Atomic Prefill 在并发变化后不重算，是否会严重影响 capacity 排序？
6. 有限轨迹 time compression 的 capacity 是否适合作为 deployment search score？
7. 二分搜索依赖 feasibility 对 `lambda` 单调，这个假设在离散事件和有限 workload 下是否可靠？
8. 目前两个 Pipeline 的对照是否过于简单或带有结论构造性？下一组最小实验应如何设计？
9. 这个 Evaluator 相比现有 network-aware Layer-Level deployment、SFC admission control、network calculus 或 QoS resource reservation，有没有潜在的新意？
10. 在与导师第一次交流前，应该把研究主张收缩到什么程度，避免过度声称？

---

## 12. 可复现方式

项目根目录执行：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sla_aware_mvp.helix_demo
```

当前测试数：

```text
12 tests
all passed
```

关键代码：

```text
src/sla_aware_mvp/domain.py       # Pipeline、Request、SLA、Violation 数据模型
src/sla_aware_mvp/evaluator.py    # 事件推进、资源记账和红线判断
src/sla_aware_mvp/capacity.py     # workload scaling 和 capacity 搜索
src/sla_aware_mvp/helix.py        # HELIX profile adapter
src/sla_aware_mvp/workload.py     # 受限 pickle loader 和 Azure-derived workload
src/sla_aware_mvp/helix_demo.py   # 本文实验配置
tests/test_helix.py               # HELIX 单位、插值、层缩放和 workload 测试
outputs/helix_mvp_result.json     # 原始结果摘要
```

---

## 13. 一句话总结

本 MVP 使用 HELIX 的 LLaMA-2 70B/A100 实测 per-layer profile 和 Azure-derived finite workload，在相同 8×A100、相同 Layer 切分和相同 SLA 下，仅改变 Stage 物理路径，得到 10 Gbps placement 约为 2.5 Gbps placement 的 2.86 倍 Conservative SLA-safe request capacity，并正确定位首次网络 violation；该结果支持 Evaluator 具有最小区分能力和可解释性，但只有在后续证明它与独立 reference serving evaluation 的 Pipeline 排序高度相关后，才能说明它作为 deployment search oracle 具有真正研究价值。
