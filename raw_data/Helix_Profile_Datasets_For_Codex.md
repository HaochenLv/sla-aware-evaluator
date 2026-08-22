# HELIX Profiling / Workload 数据源包（给 Codex）

> 用途：为当前 **固定 Layer-Level pipeline 的 Conservative SLA-Safe Evaluator** 准备可复用的数据源、下载入口、数据语义与实现约束。
>
> 本文档优先级不是“搜集越多越好”，而是：
>
> 1. 先把 **HELIX 官方 profiling 数据**接入，得到一个可工作的基线；
> 2. 再接入 **HELIX / Azure / BurstGPT workload**；
> 3. 对当前研究使用的 **Qwen2.5 + L4/L40S/A100 等组合**，用官方模型结构 + 公开硬件参数 + 可获得的现代 profiling 数据进行补充；
> 4. 所有“实测 / 代理 / 合成 / 插值”数据必须带来源标签，禁止混为一谈。

---

## 0. 最重要的结论

HELIX 官方 artifact **确实包含可直接使用的 GPU compute profiling 数据**，不是只有论文里的几张图。

其核心目录为：

```text
simulator/model_manager/
├── llama1_30b/
│   ├── a100/
│   ├── l4/
│   ├── l4x2/
│   ├── t4/
│   ├── t4x2/
│   ├── t4x4/
│   └── v100/
└── llama2_70b/
    ├── a100/
    ├── l4/
    ├── l4x2/
    ├── t4/
    ├── t4x2/
    ├── t4x4/
    └── v100/
```

每个 model × machine 目录的关键文件是：

```text
prompt_bs2time.csv
decode_bs2time.csv
<model>_<machine>.py
```

例如：

```text
simulator/model_manager/llama2_70b/a100/prompt_bs2time.csv
simulator/model_manager/llama2_70b/a100/decode_bs2time.csv
simulator/model_manager/llama2_70b/a100/llama2_70b_a100.py
```

**HELIX 代码明确把这些 profiling result 定义为“一个模型 layer 在该 machine 上运行的 profiling results”。**

CSV 第二列的时间单位是 **ms**；HELIX 读取时乘 `MilliSec = 0.001` 转为秒。

但需要特别注意：

> HELIX 的这套 profile **不是**我们最终想要的完整  
> `f_D(L_ctx, N_P, N_D, pipeline, hardware)`。
>
> 它主要是一个 **per-layer lookup table**：根据 prompt/decode 侧的 token/batch 规模查时间，线性插值后再乘以节点上承载的 layer 数。
>
> 因此它非常适合做 **第一版公开数据基线**，但不能伪装成已经刻画了 Decode context growth、Prefill/Decode 混部干扰、真实 scheduler contention 的 profile。

---

# 1. HELIX 官方入口

## 1.1 GitHub artifact

- Repository  
  https://github.com/Thesys-lab/Helix-ASPLOS25

建议 Codex 直接 clone：

```bash
git clone https://github.com/Thesys-lab/Helix-ASPLOS25.git
```

固定一个 commit hash 后再做实验，避免未来上游变化：

```bash
cd Helix-ASPLOS25
git rev-parse HEAD
```

记录到：

```text
third_party/helix/COMMIT
```

---

## 1.2 Zenodo artifact

HELIX 官方 artifact 也发布在 Zenodo，可作为论文复现时更稳定的归档来源：

- HELIX artifact v2  
  https://zenodo.org/records/14060580

建议：

- GitHub 用于开发；
- Zenodo URL + version 用于论文 artifact provenance。

---

## 1.3 Paper

- HELIX paper PDF  
  https://www.pdl.cmu.edu/PDL-FTP/BigLearning/helix.pdf

论文中说明 HELIX 会对 compute node 做 one-time profiling 来获得 token processing throughput，并对网络连接进行 one-time profiling / average bandwidth 建模。

---

# 2. HELIX 官方 GPU Compute Profiling 数据

## 2.1 LLaMA-2 70B

支持的 machine profile 目录：

```text
simulator/model_manager/llama2_70b/a100/
simulator/model_manager/llama2_70b/l4/
simulator/model_manager/llama2_70b/l4x2/
simulator/model_manager/llama2_70b/t4/
simulator/model_manager/llama2_70b/t4x2/
simulator/model_manager/llama2_70b/t4x4/
simulator/model_manager/llama2_70b/v100/
```

每个目录都应优先读取：

```text
prompt_bs2time.csv
decode_bs2time.csv
llama2_70b_<machine>.py
```

### Raw URL 模板

```text
https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/llama2_70b/<machine>/prompt_bs2time.csv

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/llama2_70b/<machine>/decode_bs2time.csv

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/llama2_70b/<machine>/llama2_70b_<machine>.py
```

其中 `<machine>`：

```text
a100
l4
l4x2
t4
t4x2
t4x4
v100
```

---

## 2.2 LLaMA-1 30B

目录：

```text
simulator/model_manager/llama1_30b/a100/
simulator/model_manager/llama1_30b/l4/
simulator/model_manager/llama1_30b/l4x2/
simulator/model_manager/llama1_30b/t4/
simulator/model_manager/llama1_30b/t4x2/
simulator/model_manager/llama1_30b/t4x4/
simulator/model_manager/llama1_30b/v100/
```

Raw URL 模板：

```text
https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/llama1_30b/<machine>/prompt_bs2time.csv

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/llama1_30b/<machine>/decode_bs2time.csv

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/llama1_30b/<machine>/llama1_30b_<machine>.py
```

---

# 3. HELIX CSV 的真实语义

## 3.1 `prompt_bs2time.csv`

示例：LLaMA-2 70B + A100 40GB。

原文件：

```text
simulator/model_manager/llama2_70b/a100/prompt_bs2time.csv
```

Raw：

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/llama2_70b/a100/prompt_bs2time.csv

文件类似：

```csv
0,0
250,2
500,4
750,5
1000,8
...
11250,77
```

解释：

```text
x = prompt 侧聚合 token / batch-size-like profiling point
y = 单 layer execution time，单位 ms
```

HELIX 在两个 profiling point 之间采用 linear interpolation。

---

## 3.2 `decode_bs2time.csv`

原文件：

```text
simulator/model_manager/llama2_70b/a100/decode_bs2time.csv
```

Raw：

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/llama2_70b/a100/decode_bs2time.csv

示例片段：

```csv
0,0
2,1.4
4,1.4
6,1.4
...
100,2.5
...
400,6
...
800,11
```

解释：

```text
x = decode 侧 profiling token / active-token batch point
y = 单 layer execution time，单位 ms
```

**注意：不要直接把 x 命名为 context length。**

HELIX 的 profile 接口和代码使用的是 `decode_num_tokens` / `decode_bs2time` 语义；它不等价于我们 evaluator 中需要的：

```text
context_length
active_decode_requests
active_prefill_requests
```

三维/多维性能曲面。

---

## 3.3 单位确认

HELIX：

```python
def load_profile_csv(file_name):
    ...
    data[int(row[0])] = float(row[1]) * MilliSec
```

并定义：

```python
MilliSec = 0.001
```

所以：

```text
CSV 第二列原值 = milliseconds
内部 simulator = seconds
```

Codex 接入时建议原始表保留 `time_ms_per_layer`，不要在落盘时偷偷变单位。

---

# 4. HELIX model-machine Python 文件里还藏着哪些“数据”

除了两个 CSV，以下文件非常有价值：

```text
simulator/model_manager/llama2_70b/<gpu>/llama2_70b_<gpu>.py
simulator/model_manager/llama1_30b/<gpu>/llama1_30b_<gpu>.py
```

里面通常包括：

```text
max_num_layers
vllm_num_blocks_dict
prompt_max_requests_dict
decode_max_tokens_dict
kv_cache_capacity
```

以 LLaMA-2 70B + A100 为例，代码中：

```text
machine_name = "A100"
max_num_layers = 12
```

并保存了：

```text
num_layers -> vLLM num_blocks
num_layers -> prompt_max_requests
num_layers -> decode_max_tokens
```

HELIX 用：

```text
kv_cache_capacity
= VLLM_BLOCK_SIZE × num_blocks × num_layers
```

生成 KV capacity。

这些数据对我们的 **memory red line** 很有参考意义。

但是：

> `vllm_num_blocks_dict` 是 HELIX 对特定模型 / 软件栈 / GPU / 配置的工程 profiling 结果，
> 不应未经标记直接套到 Qwen2.5。

---

# 5. HELIX profiling 读取逻辑——Codex 必须复现

关键文件：

```text
simulator/event_simulator/utils.py
```

Raw：

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/event_simulator/utils.py

以及：

```text
simulator/model_manager/llama2_70b/a100/llama2_70b_a100.py
```

Raw：

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/llama2_70b/a100/llama2_70b_a100.py

HELIX 基本逻辑：

```text
1. 读取 prompt_bs2time / decode_bs2time
2. 找到左右两个 profiling point
3. linear interpolation
4. 得到单 layer 时间
5. × num_on_node_layers
6. 得到该 stage 的近似计算时间
```

因此我们第一版 adapter 可以严格实现：

```python
HelixLayerProfile.lookup_prompt(x)
HelixLayerProfile.lookup_decode(x)
HelixLayerProfile.stage_time(num_layers, x)
```

而不是先改造 HELIX 数据语义。

---

# 6. HELIX 官方 Machine / VRAM / NIC Profile

## 6.1 `google_machine_profiles.ini`

路径：

```text
simulator/model_manager/google_machine_profiles.ini
```

Raw：

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/model_manager/google_machine_profiles.ini

HELIX 标注为：

```text
Real machine profiles on Google Cloud
```

包含：

| Machine | VRAM | NIC in/out |
|---|---:|---:|
| T4 | 16 GB | 10 Gbps |
| T4x2 | 32 GB | 10 Gbps |
| T4x4 | 64 GB | 10 Gbps |
| L4 | 24 GB | 10 Gbps |
| L4x2 | 48 GB | 10 Gbps |
| V100 | 16 GB | 10 Gbps |
| A100 | 40 GB | 10 Gbps |

另有：

```text
disk_speed = 1 Gbps
```

HELIX simulation example 还有一份：

```text
examples/simulation/config/machine_profile.ini
```

Raw：

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/examples/simulation/config/machine_profile.ini

---

# 7. HELIX Network Profile / Cross-Region Measurement

HELIX paper 还给出跨 Google Compute Engine region 的实际 `iperf3` bandwidth 测量，用于 geo-distributed 模拟配置。

论文中 Table 7 的量级约为：

```text
~54 Mbps – 204 Mbps
```

属于跨 region 场景，不要和单 cluster 10 Gbps NIC 配置混淆。

来源：

- paper  
  https://www.pdl.cmu.edu/PDL-FTP/BigLearning/helix.pdf

如果当前 evaluator 先做固定 edge topology，可以：

```text
P0: 使用 machine_profile.ini 的 10 Gbps 作为 HELIX baseline
P1: 使用 Table 7 的 cross-region Mbps 数据做 network sensitivity
P2: 再换成我们自己的 1/10/25/100 Gbps edge-link family
```

---

# 8. HELIX 自带 Workload / Trace 数据

> 这些不是 GPU compute profile，但当前 evaluator 搜索最大安全 arrival rate 时非常有用。

HELIX 目录：

```text
simulator/trace_generator/
```

其中：

```text
arrival_rate/
length_data/
arrival_rate_sampler.py
length_sampler.py
trace_generator.py
simulator_query_feeder.py
```

---

## 8.1 Arrival traces

目录：

```text
simulator/trace_generator/arrival_rate/
```

包含：

```text
azure_code_arrive_time.pkl
azure_conv_arrive_time.pkl
```

即：

- Azure Code
- Azure Conversation

Raw URL：

```text
https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/trace_generator/arrival_rate/azure_code_arrive_time.pkl

https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/trace_generator/arrival_rate/azure_conv_arrive_time.pkl
```

---

## 8.2 Request input/output length distributions

目录：

```text
simulator/trace_generator/length_data/
```

HELIX artifact 内可见：

```text
alpaca_input.pkl
alpaca_output.pkl

azure_code_input.pkl
azure_code_output.pkl

azure_conv_input.pkl
azure_conv_output.pkl

shared_gpt_input.pkl
shared_gpt_output.pkl
```

Raw URL 模板：

```text
https://raw.githubusercontent.com/Thesys-lab/Helix-ASPLOS25/master/simulator/trace_generator/length_data/<filename>
```

这意味着 HELIX 已经帮我们准备了：

| Dataset | Arrival | Input length | Output length |
|---|---:|---:|---:|
| Azure Conversation | ✓ | ✓ | ✓ |
| Azure Code | ✓ | ✓ | ✓ |
| Alpaca | — | ✓ | ✓ |
| ShareGPT | — | ✓ | ✓ |

其中 Alpaca / ShareGPT 在 HELIX 中主要作为 length distribution 来源，不要声称它们本身提供 HELIX 使用的真实 request arrival timestamps。

---

# 9. HELIX paper 主实验使用的 workload

HELIX paper 主实验使用的是：

```text
Azure Conversation trace
```

来源链：

```text
Azure LLM inference traces
      ↓
Splitwise
      ↓
HELIX trace preprocessing
```

HELIX paper 对 trace 做了过滤：

```text
input length  > 2048  -> prune
output length > 1024  -> prune
```

论文报告过滤后约：

```text
16,657 requests
average input ≈ 763 tokens
average output ≈ 232 tokens
```

HELIX simulator 代码中的 dataset constants 也非常接近：

```python
AVG_INPUT_LEN = 750
AVG_OUTPUT_LEN = 240
MAX_INPUT_LEN = 2048
```

这些常量在：

```text
simulator/event_simulator/utils.py
```

**建议：我们的 evaluator 不要硬编码这些均值。**
把它们仅作为 HELIX baseline validation 使用。

---

# 10. Azure LLM Inference Dataset——上游真实来源

## 10.1 Azure Public Dataset repository

官方入口：

https://github.com/Azure/AzurePublicDataset

2023 LLM inference dataset 说明：

```text
AzureLLMInferenceDataset2023.md
```

该数据被 Splitwise 使用。

典型字段：

```text
TIMESTAMP
ContextTokens
GeneratedTokens
```

这正好可以映射到我们每个 request：

```text
t_arr
L_in
L_out
```

推荐 normalized schema：

```csv
request_id,arrival_time_s,input_tokens,output_tokens,source
0,0.000,512,128,azure_2023
...
```

---

## 10.2 Splitwise repo

Splitwise 是 Azure 2023 trace 和 HELIX workload 之间非常有价值的参考实现。

建议 Codex 同时保存其 trace parser / generator 逻辑作为对照。

搜索入口：

https://github.com/Mutinifni/splitwise-sim

如果仓库组织发生变化，优先从 Azure 2023 dataset README 中提供的 Splitwise reference 反查。

---

# 11. BurstGPT——强烈建议作为当前项目的第二套真实 workload

官方 GitHub：

https://github.com/HPMLL/BurstGPT

官方 releases：

https://github.com/HPMLL/BurstGPT/releases

BurstGPT 是真实 LLM serving workload trace，当前公开版本包含数百万请求，字段包括：

```text
Timestamp
Model
Request tokens
Response tokens
Total tokens
Log Type
```

新版还加入：

```text
Session ID
Elapsed time
```

对于当前 evaluator，最重要的是：

```text
Timestamp        -> arrival time
Request tokens   -> L_in
Response tokens  -> L_out
```

它很适合做：

```text
finite workload trajectory
burstiness
arrival-rate scaling
input/output length heterogeneity
```

建议第一版只取一小段 window：

```text
5 min / 10 min / 30 min
```

然后通过 time compression / dilation 生成：

```text
W_lambda = G(W0, lambda)
```

不要一开始把几百万条全部塞进 evaluator。

---

# 12. DistServe 公开 profiling——作为“现代补充”，不是 HELIX 数据

HELIX 的 profile 最大缺口是：

```text
没有显式刻画 context length 对 decode per-token time 的增长
没有显式刻画 N_P / N_D 混部状态
```

因此可以补充 DistServe 的 profiling 资源。

## 12.1 DistServe official repo

https://github.com/LLMServe/DistServe

DistServe 本身就是 TTFT / TPOT + Prefill/Decode disaggregation 方向，和我们的 SLA evaluator 的性能建模更接近。

其代码包含 profiling infrastructure / profiling database 相关实现，可作为：

```text
ProfileProvider interface
数据字段设计
Prefill / Decode 分离
memory profiling
```

的参考。

---

## 12.2 DistServe public profiling dataset on Hugging Face

可检查：

https://huggingface.co/datasets/DistServe/2025-05-06T14-automatic-profiling

当前公开快照可见约：

```text
44.5 MB
H100_llama8b_pp1_tp1/
```

用途：

```text
P1 supplementary measured profile
```

但必须加：

```text
source = distserve_public_profile
model = llama8b
gpu = H100
parallelism = pp1_tp1
```

**禁止**因为它比 HELIX 新，就把它直接缩放后说成 Qwen2.5/L4 的真实数据。

---

# 13. 当前研究模型结构数据：Qwen2.5 官方 Hugging Face config

当前部署研究使用 Qwen2.5 7B / 14B / 32B / 72B 时，模型结构不要猜。

官方 collection：

https://huggingface.co/collections/Qwen/qwen25

建议 Codex 直接通过 Hugging Face `config.json` 读取：

```text
num_hidden_layers
hidden_size
intermediate_size
num_attention_heads
num_key_value_heads
torch_dtype
max_position_embeddings
```

---

## 13.1 Qwen2.5-7B

https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json

关键结构：

```text
num_hidden_layers = 28
hidden_size = 3584
num_attention_heads = 28
num_key_value_heads = 4
```

---

## 13.2 Qwen2.5-14B

https://huggingface.co/Qwen/Qwen2.5-14B/blob/main/config.json

典型结构：

```text
num_hidden_layers = 48
hidden_size = 5120
num_attention_heads = 40
num_key_value_heads = 8
```

---

## 13.3 Qwen2.5-32B

官方目标 repo：

```text
Qwen/Qwen2.5-32B
```

Config：

https://huggingface.co/Qwen/Qwen2.5-32B/resolve/main/config.json

---

## 13.4 Qwen2.5-72B

https://huggingface.co/Qwen/Qwen2.5-72B/blob/main/config.json

关键结构：

```text
num_hidden_layers = 80
hidden_size = 8192
num_attention_heads = 64
num_key_value_heads = 8
```

---

# 14. NVIDIA 官方硬件规格——用于 analytical / synthetic profile fallback

如果没有目标 GPU × Qwen 的实测 profile，当前 evaluator 允许使用：

```text
Measured profile
or
Estimated profile
```

因此应建立 `hardware_specs.csv`，字段至少：

```csv
gpu_type,vram_gib,mem_bw_gbps,fp16_or_bf16_tflops,source
```

建议官方来源：

```text
NVIDIA T4
NVIDIA V100
NVIDIA L4
NVIDIA L40S
NVIDIA A100
NVIDIA H100
Jetson AGX Orin
```

入口：

- NVIDIA Data Center GPUs  
  https://www.nvidia.com/en-us/data-center/

- NVIDIA L4  
  https://www.nvidia.com/en-us/data-center/l4/

- NVIDIA L40S  
  https://www.nvidia.com/en-us/data-center/l40s/

- NVIDIA A100  
  https://www.nvidia.com/en-us/data-center/a100/

- NVIDIA H100  
  https://www.nvidia.com/en-us/data-center/h100/

- NVIDIA Jetson Orin  
  https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/

如果具体网页改版，优先下载 NVIDIA 官方 datasheet PDF 并在 metadata 中保存 PDF 名称和访问日期。

---

# 15. 推荐给我们项目的统一数据目录

Codex 不要让外部项目的数据格式污染 evaluator 核心代码。

推荐：

```text
data/
├── raw/
│   ├── helix/
│   │   ├── compute_profiles/
│   │   ├── machine_profiles/
│   │   └── traces/
│   ├── azure/
│   ├── burstgpt/
│   ├── distserve/
│   ├── qwen_configs/
│   └── hardware_specs/
│
├── processed/
│   ├── compute_profile.csv
│   ├── memory_profile.csv
│   ├── hardware_profile.csv
│   ├── network_profile.csv
│   └── workload/
│       ├── azure_conv.csv
│       ├── azure_code.csv
│       └── burstgpt_window_*.csv
│
└── provenance/
    ├── sources.yaml
    └── checksums.sha256
```

---

# 16. 我们自己的统一 profile schema

## 16.1 `compute_profile.csv`

不要强行把所有来源转成一个虚假的多维 profile。

建议 long-table：

```csv
profile_id,source,model,gpu_type,phase,x_kind,x_value,time_ms_per_layer,num_layers_basis,context_len,n_prefill,n_decode,parallelism,measured
```

例如 HELIX：

```csv
helix_llama2_70b_a100_d_100,helix,llama2_70b,A100,decode,decode_tokens,100,2.5,1,,,,,true
```

HELIX 中不存在的维度：

```text
context_len = NA
n_prefill = NA
n_decode = NA
```

**不要填 0。**

0 会被误解为“实测时没有并发”，NA 才表示“该数据源没有这个维度”。

---

## 16.2 `memory_profile.csv`

```csv
source,model,gpu_type,num_layers,max_num_layers,vllm_blocks,kv_token_capacity,vram_gib,measured
```

HELIX model-machine Python 文件中可解析：

```text
max_num_layers
vllm_num_blocks_dict
kv_cache_capacity
```

---

## 16.3 `hardware_profile.csv`

```csv
source,gpu_type,vram_gib,mem_bw_gbps,fp16_tflops,bf16_tflops,nic_in_gbps,nic_out_gbps
```

数据优先级：

```text
NVIDIA official datasheet
>
HELIX machine profile
>
third-party summary
```

---

## 16.4 `workload.csv`

统一为：

```csv
request_id,arrival_time_s,input_tokens,output_tokens,source,original_id
```

所有 evaluator 只认这一层。

---

## 16.5 `network_profile.csv`

```csv
source,src,dst,bandwidth_mbps,latency_ms,measurement_method,notes
```

对于 HELIX：

```text
single-cluster 10 Gbps baseline
cross-region iperf3 measurements
```

必须分开保存。

---

# 17. 给当前 Evaluator 的 ProfileProvider 接口建议

第一版实现：

```python
class ProfileProvider:
    def prefill_time(
        self,
        *,
        model,
        gpu_type,
        num_layers,
        input_tokens,
        n_prefill=None,
        n_decode=None,
    ) -> float:
        ...

    def decode_time_per_token(
        self,
        *,
        model,
        gpu_type,
        num_layers,
        context_len,
        n_prefill=None,
        n_decode=None,
    ) -> float:
        ...
```

具体 adapter：

```text
HelixProfileProvider
DistServeProfileProvider
AnalyticalProfileProvider
SyntheticCalibratedProfileProvider
```

---

# 18. HELIX adapter 的正确做法

HELIX 原始数据没有完整的：

```text
context_len
n_prefill
n_decode
```

所以第一版不要假装有。

建议：

```python
class HelixProfileProvider(ProfileProvider):
    """
    Public-profile baseline.

    Data semantics:
    - per-layer prompt/decode lookup tables
    - linear interpolation
    - layer scaling
    - no explicit context/interference dimensions
    """
```

输出除了数值，还应带 provenance：

```python
ProfileValue(
    seconds=...,
    source="helix",
    measured=True,
    dimensions_observed={
        "phase": True,
        "aggregate_tokens": True,
        "context_len": False,
        "n_prefill": False,
        "n_decode": False,
    },
)
```

这样未来替换成 context-aware profile 时 evaluator 核心无需改。

---

# 19. 对 Decode context growth 的处理：当前最容易犯错的地方

我们的 Evaluator 需要：

```text
L_ctx ↑
→ T_D ↑
→ TPOT 剩余网络预算 ↓
→ required bandwidth ↑
```

HELIX 原始 `decode_bs2time.csv` **不能直接证明这条 context-dependent 曲线**。

因此可选三种实现层次：

### Level A：HELIX-only baseline

```text
T_D = HELIX lookup(batch/decode tokens) × num_layers
```

优点：

```text
公开、可复现、简单
```

缺点：

```text
没有 context growth
```

实验中必须写：

```text
context-independent compute-profile baseline
```

---

### Level B：HELIX + analytical context correction

构造：

```text
T_D(ctx)
=
T_HELIX
×
g(ctx)
```

其中 `g(ctx)` 来自 attention/KV memory-access analytical model。

这属于：

```text
synthetic / analytical calibrated profile
```

不是 measured HELIX profile。

---

### Level C：真正 context-aware profiling

未来自己或通过公开数据得到：

```text
(model, gpu, num_layers, batch, context_len, phase)
→ time
```

再进一步加入：

```text
N_P
N_D
```

这才是 evaluator 最终理想接口。

---

# 20. 不要把 HELIX profile 直接当成 Qwen2.5 profile

这是给 Codex 的硬约束：

```text
LLaMA-2 70B + A100 profile
!=
Qwen2.5-72B + A100 profile
```

可以做：

```text
proxy baseline
scaled synthetic estimate
sanity check
```

但必须写：

```text
source_model = llama2_70b
target_model = qwen2.5_72b
profile_type = proxy
```

不能写成：

```text
measured qwen2.5 profile
```

---

# 21. 推荐的最小可用数据集组合

如果目标是尽快把 evaluator 跑起来，不要一次吞下所有数据。

## P0：必须

```text
HELIX repo
HELIX LLaMA-2 70B compute profiles
HELIX machine_profile.ini
Azure Conversation trace / HELIX processed pkl
Qwen2.5 config.json
```

目标：

```text
固定 pipeline
→ workload
→ HELIX-style compute lookup
→ event-driven evaluator
→ conservative SLA-safe capacity
```

---

## P1：紧接着加

```text
HELIX LLaMA-1 30B profiles
BurstGPT
Azure Code
DistServe public profiling
NVIDIA official hardware specs
```

用途：

```text
cross-model sensitivity
workload sensitivity
burstiness
profile-source sensitivity
```

---

## P2：后面再加

```text
Alpaca length distribution
ShareGPT length distribution
Azure 2024 LLM traces
更多 GPU / 模型实际 profiling
```

不要让 P2 阻塞 evaluator 最小闭环。

---

# 22. Codex 可直接执行的下载/解析任务

```text
TASK: Build the profiling/workload data layer for the current
Conservative SLA-Safe Evaluator.

1. Clone HELIX official artifact:
   https://github.com/Thesys-lab/Helix-ASPLOS25

2. Parse all compute profiles under:
   simulator/model_manager/llama1_30b/
   simulator/model_manager/llama2_70b/

3. For each GPU/model pair, ingest:
   - prompt_bs2time.csv
   - decode_bs2time.csv
   - model-machine .py metadata:
     max_num_layers,
     vllm_num_blocks_dict,
     prompt_max_requests_dict,
     decode_max_tokens_dict

4. Preserve raw semantics:
   - timing CSV second column is ms
   - timing is per layer
   - interpolate only on the original x-axis
   - do NOT reinterpret decode x-axis as context length

5. Parse:
   simulator/model_manager/google_machine_profiles.ini

6. Parse HELIX workload artifacts:
   simulator/trace_generator/arrival_rate/
   simulator/trace_generator/length_data/

7. Normalize workload to:
   request_id, arrival_time_s, input_tokens, output_tokens, source

8. Download / register:
   - Azure LLM Inference Dataset
   - BurstGPT
   - Qwen2.5 official config.json
   - optional DistServe public profiling dataset
   - NVIDIA hardware specs metadata

9. Implement adapters:
   - HelixProfileProvider
   - AnalyticalProfileProvider
   - optional DistServeProfileProvider

10. Every returned profile value must carry provenance:
    measured/proxy/synthetic,
    source model,
    source GPU,
    observed dimensions.

11. Do not alter evaluator semantics:
    - fixed Layer-Level pipeline
    - event-driven state updates
    - no token-by-token simulation
    - no new scheduling/routing/replica optimization

12. Add tests that verify:
    - HELIX CSV ms → seconds conversion
    - interpolation reproduces HELIX behavior
    - stage time scales with number of layers
    - missing context/N_P/N_D dimensions remain NA
    - workload conversion preserves arrival/input/output fields
```

---

# 23. 建议 Codex 第一轮先产出的文件

```text
src/profiling/base.py
src/profiling/helix.py
src/profiling/analytical.py
src/workload/loader.py

data/provenance/sources.yaml

tests/test_helix_profile.py
tests/test_workload_loader.py

scripts/fetch_helix_profiles.py
scripts/fetch_workloads.py
scripts/build_processed_profiles.py
```

---

# 24. `sources.yaml` 建议

```yaml
helix:
  repo: https://github.com/Thesys-lab/Helix-ASPLOS25
  zenodo: https://zenodo.org/records/14060580
  paper: https://www.pdl.cmu.edu/PDL-FTP/BigLearning/helix.pdf
  profile_type: measured_public_artifact

azure_llm_2023:
  repo: https://github.com/Azure/AzurePublicDataset
  type: real_workload_trace

burstgpt:
  repo: https://github.com/HPMLL/BurstGPT
  releases: https://github.com/HPMLL/BurstGPT/releases
  type: real_workload_trace

distserve_profile:
  repo: https://github.com/LLMServe/DistServe
  dataset: https://huggingface.co/datasets/DistServe/2025-05-06T14-automatic-profiling
  type: measured_public_profile_secondary

qwen25:
  collection: https://huggingface.co/collections/Qwen/qwen25
  type: official_model_config

nvidia:
  data_center: https://www.nvidia.com/en-us/data-center/
  type: official_hardware_spec
```

---

# 25. 我建议当前研究实际采用的路线

当前不要等待“完美的 Qwen × 所有 GPU × 所有 context × 所有并发”的公开数据集，因为大概率不存在一个刚好完全匹配我们接口的数据包。

最稳妥的实验路线：

```text
Stage 1
HELIX measured profiles
→ 验证 evaluator 代码和数学逻辑

Stage 2
HELIX + Azure/BurstGPT
→ 验证 workload / event-driven capacity search

Stage 3
Qwen2.5 official architecture
+ NVIDIA hardware specs
+ HELIX/DistServe calibration
→ 生成标注清楚的 estimated profile

Stage 4
若以后拿到真实 GPU
→ 用同一个 ProfileProvider 接口替换 estimated data
```

这样：

```text
Evaluator 本身
```

和：

```text
Profiling data quality
```

是解耦的。

即便未来换 profile，状态机、SLA→resource commitment、memory red line 和 capacity search 都不用推倒重写。

---

# 26. 最终数据可信度标签

建议所有数据统一四档：

```text
M = measured
P = public measured, but proxy model/GPU/software stack
E = estimated / analytical
S = synthetic sensitivity value
```

例如：

```text
HELIX LLaMA2-70B A100
→ M

HELIX LLaMA2-70B A100 用来近似 Qwen2.5-72B A100
→ P

Qwen2.5 FLOPs + A100 bandwidth roofline
→ E

为了 sweep 人工设置 0.8×/1.0×/1.2×
→ S
```

论文结果表和原始 CSV 都保留这个标签。

---

# 27. 一句话交给 Codex

> 先把 HELIX 的 `prompt_bs2time.csv` / `decode_bs2time.csv` 作为 **per-layer measured baseline** 严格复现，不改变其原始语义；再把 Azure/BurstGPT 统一成 finite request workload；对于 HELIX 没有的 context/N_P/N_D 维度，用独立的 analytical/secondary-profile adapter 补充，并在每个返回值上保留 measured/proxy/estimated provenance，绝不把代理数据冒充目标 Qwen/GPU 的真实 profiling。
