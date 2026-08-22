# OneRec Decoder Attention W8A8 方案

**状态：** 讨论中（Proposed）  
**日期：** 2026-08-21  
**范围：** graph3 OneRec，`beam_width=512`；暂不覆盖 1024 / 2048

本文档把「Decoder Attention 线性层做静态 W8A8」的结论、约束、仓库改动和仓库外工作整理成一份可评审材料。`enable_graph=false` 与单卡默认 `world_size=1` 已确认；其余开放问题见第 8 节。

相关文档：

- [生成式推荐设计文档](generative_recommendation_design.md)
- [Graph Mode 设计文档](graph_mode_design.md)
- [模型侧交付说明](onerec_attn_w8a8_model_handoff.md)（给模型 / 转换同学转发）

---

## 1. 背景

graph3 OneRec 当前配置：

- Encoder 4 层 dense，Decoder 12 层 MoE，`d_model=2048`，GQA 4，`head_dim=128`
- Decoder MoE 已是 `quantize: "w8a8_dynamic"`（W8 静态 + A8 动态 GMM）
- Attention 的 Q/K/V/O（Self + Cross）仍是 BF16
- `config.json` 里的 `"quantize": "w8a8_dynamic"` **只描述 MoE**，不等于 KV int8，也不等于 Attention W8A8

KV cache int8 在 graph3 + GQA + 小 decode 步数下收益很小，不作为本方案内容。

在 `beam=512` 下，decode 的有效 batch 是 `max_seqs_per_batch × beam_width`。Attention 线性层随 beam 线性增长，是 MoE 之后最大的一块仍为 BF16 的 GEMM。

### 1.1 目标运行参数

本方案按下面这组线上参数设计，验收也必须在这组参数下进行，而不是在单流满核上「先测通再假设能搬上去」。

| 参数 | 值 | 对方案的影响 |
|---|---|---|
| `beam_width` | 512 | 单流 decode `m = 2 × 512 = 1024` |
| `max_seqs_per_batch` | 2 | 同上 |
| `rec_worker_max_concurrency` | 2 | **两套完整模型、两条 NPU stream** |
| `enable_multistream_perf_mode` | true | prepare / model forward 不串行；`XLLM_ATB_ACLNN_THREAD_LOCAL_CACHE=1` |
| `enable_onerec_multistream_core_split` | true | 每条 stream 限制核数 |
| `onerec_multistream_core_ratio` | 0.5 | 每条 stream **一半 Cube / 一半 Vector** |
| `enable_onerec_async_perf` | true | 异步 beam search / cache select；`TASK_QUEUE_ENABLE` + `PER_STREAM_QUEUE` |
| `enable_onerec_prefill_acl_graph` | false | Decoder prefill **eager** |
| `enable_graph` | **false** | prefill / decode **都不进 ACL graph**，全程 eager |
| `world_size` / `tp_size` | **未显式设置，默认 1** | 单卡，不做张量并行；load 时 shard 是 no-op |

一步 decode、单流量级（graph3，12 层）：

- MoE（已 w8a8）：约 **1.55T**
- Attention QKV+O（仍 BF16）：约 **0.25T**（约 14%）
- 双流同时满载时，设备上相当于两份 `m=1024`，各用 50% 核

半核上 `2048×2048` 的 BF16 GEMM 更偏计算受限，INT8 Cube 更能对冲「核被砍半」。因此这套参数下 Attention W8A8 比「双流抢满核」更值得做。e2e 不会翻倍，因为 MoE 仍是大头。

### 1.2 成功标准

- 无 Attention 量化权重的现有 graph3：**行为与现在完全一致**
- 有量化权重：Decoder 八个 Attention 线性层走静态 W8A8，MoE 仍是 `w8a8_dynamic`
- 上述 flag 全开、`beam=512`、双流同时 decode 能跑通
- 单层 QKV/O 相对 BF16 cosine ≥ 0.99；业务 topk / 推荐指标门槛由业务侧定
- 性能先报 Attention 线性层时间，再报 e2e（预期 e2e 大约数个点到十几百分点）

---

## 2. 决策

**Decoder Attention 走静态 W8A8（与 Qwen Attention 同一条 ATB 契约），不要复用 MoE 的 `w8a8_dynamic`。**

计算图：

```text
x (BF16)
  → RMSNorm + Quant (INT8, input_scale)
  → W8A8 GEMM (INT8 × INT8)
  → dequant (deq_scale, FP32[n]) → BF16
  → XAttention / CrossAttention（仍 BF16，不动 kernel）
  → O 投影同样 W8A8
```

v1 覆盖 Decoder 每层 8 个线性层：

- Self Q/K/V（packed 成一条 GEMM）+ Self O
- Cross Q/K/V（独立，不 pack）+ Cross O

Encoder Attention / Encoder FFN 保持 BF16。

开关：**看 checkpoint 是否带 Attention `deq_scale`**，不改 `quantize: "w8a8_dynamic"` 的含义。

线性层后端：Decoder 量化后 **固定 ACLNN `QuantBatchMatmul`**，不再按 `ntokens < 128` 切 ATB。decode 的 `m=1024` 本来就 ≥ 128；第 0 轮若只有 2 个 token，也统一 ACLNN，避免半核上再养一条 ATB W8A8。

`enable_graph=false`：W8A8 **不必**考虑 ACL graph capture / replay。`onerec_acl_graph_executor` 不在 v1 改动面里。

权重：`concurrency=2` 会 load **两份**模型，两份都要是 int8 + `FRACTAL_NZ`。`world_size=1`，QKV/O 不做 TP 切分。

---

## 3. 否决的替代方案

| 方案 | 否决原因 |
|---|---|
| Attention 套 `w8a8_dynamic` | graph3 是 BF16；ATB 写明 `LINEAR_W8A8_DYNAMIC_QUANT` 不支持 BF16；FusionAttention 默认 `enableNormQuantOp=true` 会走到未实现的 `LINEAR_W8A8_DYNAMIC_DEQUANT` |
| v1 强制 ATB 线性层 | 与 `enable_multistream_perf_mode`（ACLNN thread-local cache）和 decode `m=1024` 相反；半核双流的主路径就是 ACLNN |
| 只量化 Q/O，K/V 保持 BF16 | Self / Cross 共用 `packQuantType[0]` 和 `linearDescs`。Self 一旦 `ALL_W8A8`，Cross K/V 也会按 W8A8 建图。拆开需要改 ATB，不作为 v1 |
| KV cache int8 | GQA 后 Cross KV 与 beam 无关、体积小；读写全是自定义 BF16 kernel |
| Encoder Attention / FFN W8A8 | Encoder 只跑一轮，扩 beam 几乎吃不到 |
| 加载期只量化权重、激活 scale 拍脑袋 | 静态 A8 的 `input_scale` 错了，Attention 比 MoE 更容易糊；必须用 `2×512` 校准 |
| 为 W8A8 关掉 `core_split` / `async_perf` | 方案必须在这套线上 flag 下工作 |

---

## 4. 量化契约

### 4.1 张量

对每个线性层：

| 张量 | dtype | shape | 说明 |
|---|---|---|---|
| `weight` | int8 | `[n, k]`，`transposeB=true` | per-channel：`s_w = absmax(\|W[n,:]\|) / 127` |
| `input_scale` | BF16 | `[1]` | RMSNorm 输出 absmax，校准得到 |
| `input_offset` | 多为 int8 0 | `[1]` | 与 Qwen W8A8 对齐 |
| `deq_scale` | **FP32** | `[n]` | `input_scale × s_w` |

`deq_scale` 必须是 FP32：这是 Ascend W8A8 在 **BF16 输出** 下对 `in_descale` 的硬件/ATB 契约（FP16 模型则是 int64）。OneRec loader 已有「BF16 模型不要把 `deq_scale` 转成 BF16」的特例。不要和 `input_scale` 混淆：前者是 GEMM 后 per-channel 反量化，后者是 RMSNorm 量化用的标量。

OneRec 无 Attention bias，bias 槽保持 placeholder。

### 4.2 图参数（仅 Decoder，且检测到量化权重时）

```text
packQuantType[0]  = ALL_W8A8
linearDescs       = {W8A8_PER_TENSOR × 4}     # Q, K, V, O
linearQuantType   = {INT, INVALID, INVALID, INT, ...}
matmulBackend     = ACLNN
enableNormQuantOp = true                      # 已是 FusionAttention 默认
kvQuant           = false
```

MoE：`moePackQuantType = ALL_W8A8_DYNAMIC` 不变。

Self packed 与 Cross unpacked 可以共存：`CheckPack(ALL_W8A8)=true` 只影响 Self 的 `QKVLinearSplitPack`；Cross 一直是三条独立 `NormLinear`，但会读同一套 `linearDescs`，因此 Cross Q/K/V/O 也必须是 int8。

RMSNorm 的 INT8 quant param 在 `block_layer.cpp` 里已经接好，当前只是因为 `linearDescs` 为 BF16 而没有走到量化路径。**预期 ATB `block_layer.cpp` 不用改。**

### 4.3 权重命名（需能被现有 suffix 匹配扫到）

```text
decoder.block_xx.layer.0.SelfAttention.{q,k,v,o}.weight
decoder.block_xx.layer.0.SelfAttention.{q,k,v,o}.input_scale
decoder.block_xx.layer.0.SelfAttention.{q,k,v,o}.input_offset
decoder.block_xx.layer.0.SelfAttention.{q,k,v,o}.deq_scale
decoder.block_xx.layer.1.EncDecAttention.{q,k,v,o}.*   # 同上四类
```

Self：`cat(Q,K,V)` 权重和 `deq_scale`；Q/K/V 共用同一 RMSNorm，`input_scale` 应相同，packed 后只留一份。  
Cross：**不要 cat**（现有注释已要求）。  
NZ：跟 **Qwen Attention + ACLNN QuantBatchMatmul**，不要抄 MoE GMM 的 `[E,K,N]` 布局。

---

## 5. 本仓库代码改动

### 5.1 必改

主战场：`xllm/core/layers/npu/npu_onerec_block_layer_impl.cpp` / `.h`。

构造时 `param_from_args` 仍按 BF16。`merge_loaded_weights` 在 `init_layer` 之前检测 Decoder `Q.weight` 是否为 int8（或 `q.deq_scale` 非 placeholder），再翻转第 4.2 节的图参数。Encoder 永远不打开。

| 点 | 做法 |
|---|---|
| 权重 mapping | 只给 `kOneRecDecoderWeightMapping` 加 scale/offset/deq_scale；MoE mapping 从它 copy，会自动带上。Encoder mapping 不动 |
| TP shard | 线上 `world_size=1`，现有 `get_sharded_tensor` 不会切。映射可继续沿用 Qwen 约定（Q/K/V `weight`/`deq_scale` dim 0，O `weight` dim 1，O `deq_scale` 与 `input_scale`/`offset` 不切），以便以后开 TP 不返工；v1 不验收 `world_size>1` |
| `verify_loaded_weights` | scale 类张量 **默认允许 placeholder**（否则现有 BF16 graph3 会挂）；一旦判定 W8A8 则 8 组 scale 必填 |
| `load_state_dict` | 确认 `deq_scale` 保持 FP32、`input_scale` 保持 BF16、`input_offset` 不被转成 BF16；不要把 MoE `weight_scale` 误绑到 Attention 槽 |
| `merge_loaded_weights` | Self cat 权重 + `deq_scale`；Cross 不 cat；int8 权重 `FRACTAL_NZ` |
| `init_layer` / `forward` | Attention W8A8 时 `use_atb_small_tokens` 恒为 false，不再切 `prefill_node_atb_` |
| 日志 | 每层打 `attn_w8a8=on/off`、Q 权重 dtype、`matmulBackend`，确认两条 pipeline 都 load 对了 |

头文件加 Decoder 侧标志（例如 `attn_w8a8_enabled_`）。不必改公开 C API。

建议补测试（仓库里目前没有 OneRec layer 单测）：

1. BF16 ckpt 无 scale → 不 FATAL，仍 BF16 图
2. 假 int8 + `deq_scale` → Self packed 形状、Cross 不 cat、`deq_scale` 仍 FP32
3. W8A8 时不再走 ATB 小 token 节点

### 5.2 视情况才改

| 文件 | 触发条件 | 改什么 |
|---|---|---|
| `third_party/xllm_atb_layers/models/onerec/layer/block_layer.cpp` | 建图失败（Cross 被当成 packed，或 NormQuant 未生效） | 给 Cross 单独 `packQuantType` / `linearDescs`；v1 尽量不碰 |
| `quant_args` / `config.json` | 权重检测不够、要显式开关 | 可选 `quantize_attn`；默认仍用权重检测 |

`rec_worker_impl.cpp` 预期不用改：`core_split` / `async_perf` / 双 pipeline load 已经存在。

### 5.3 明确不改

XAttention / CrossAttention kernel、`select_unshared_kv`、beam search、MoE GMM、Encoder 层、KV cache dtype、`graph3/config.json` 的 `"quantize": "w8a8_dynamic"`、`onerec_acl_graph_executor.cpp`（`enable_graph=false`）。

---

## 6. 仓库外必须做

### 6.1 导出 Decoder Attention W8A8 权重（阻塞项）

现有 graph3 Attention 是 BF16，没有 `deq_scale`，不改代码也跑不起来 W8A8。给模型 / 转换同学的可转发说明见 [模型侧交付说明](onerec_attn_w8a8_model_handoff.md)。

离线 PTQ（脚本可放 `tools/`，也可用现有量化工具链）：

1. 对 Decoder 8 个矩阵做 per-channel 权重量化
2. 用 **2 条请求 × beam=512** 的真实 rec 流量校准 RMSNorm 输出 → `input_scale`
3. 计算 FP32 `deq_scale`
4. MoE 的 w8a8_dynamic 原样拷贝；Encoder Attention 保持 BF16

不建议加载期「只量化权重、激活 scale 拍脑袋」。

### 6.2 数值对齐

- 单层：同一输入 BF16 vs W8A8，Q/K/V/O cosine ≥ 0.99
- 端到端：beam=512 校准/业务集上的 topk 重叠与推荐指标
- 掉点先查校准是否用了大 beam，再考虑只量化 O（需要改 ATB 解耦，不是 v1）

### 6.3 性能与双流验收

必测组合即第 1.1 节全表 flag。重点：

1. 两条 pipeline 日志都是 `attn_w8a8=on`
2. **半核** 下 QuantBatchMatmul 是否按 stream resource limit tiling（不要按整卡核数）
3. 双流同时 decode + 异步 beam/cache select：无串台、无全局 ACLNN cache 冲突
4. 对照：`conc=1` 满核 vs `conc=2` 半核

W8A8 算子必须走 thread-local ACLNN cache 和 per-stream queue。不要为量化关掉 `core_split` / `async_perf`。

量化 ckpt 与 BF16 ckpt 需要可回滚：无 `deq_scale` 应自动回 BF16 Attention。

---

## 7. 分期

1. **PTQ 脚本 + 一层假权重**：先有 int8/`deq_scale` 文件  
2. **mapping / load / verify / merge / NZ / 图参数翻转 / 取消 ATB 小 token 切分**  
3. **`conc=1` 打通 QuantBatchMatmul**  
4. **八层 + 第 1.1 节 flag 全开**（半核双流异步、eager、单卡）  
5. 校准精度 + 线性层 / e2e profile

---

## 8. 已确认项与仍待讨论的问题

### 已确认

- **`enable_graph=false`：** prefill / decode 均为 eager，W8A8 不进 ACL graph，不必改 `onerec_acl_graph_executor`。
- **`world_size` 未显式设置：** 使用默认值 1（`Options::world_size` 默认 1，启动脚本 `NNODES=1`、单 device）。v1 按单卡实现和验收，不做 TP。

### 仍待讨论

1. **精度门槛：** 单层 cosine 0.99 是否够？业务 topk 重叠 / CTR 代理指标由谁定、用哪份校准集？
2. **PTQ 责任：** 权重导出是推理侧脚本，还是训练/转换流水线产出？命名是否必须严格按第 4.3 节？
3. **显式开关：** 仅权重检测是否够？是否要 `quantize_attn` 防止误 load？
4. **e2e 预期：** Attention 线性层约 14% 算力，半核下墙钟收益可能高于 14%，但仍远小于 MoE。是否接受「线性层明显加速、e2e 只有数个点到十几百分点」作为成功？

---

## 9. 非目标（本轮不做）

> 更新：`beam_width` 1024 / 2048 已在后续改动中放开，见第 9.1 节；
> 「`beam_width == top_k`」契约经验证无法解除，原因见第 9.2 节。本节其余条目仍然成立。

- KV cache int8、softmax int8、Encoder W8A8
- Attention 走 `w8a8_dynamic`
- XAttention `qNBlockTile`、lm_head INT8、融合约束 topk（不物化全表 logits）
- ACL graph capture（`enable_graph=false`）
- 张量并行（`world_size=1`）

这些在更大 beam 下会重新变成问题，但不阻塞本方案。

### 9.1 已放开：beam_width 1024 / 2048

- `kRecConstrainedTopKMaxK` 512 → 1024（UB 约 132KB / 192KB）。
- fused 门控 `kRecConstrainedTopKFusedMaxK` 同步抬到 1024；超过上限的请求回落到
  composite 选择器而不是报错，所以 `beam_width=2048` 仍可跑通，只是三轮都走 composite。
- API 层 `top_logprobs` 的硬上限从 2000 放宽到 `max(2000, beam_width)`，
  否则 `beam_width=2048` 的请求在参数校验阶段就会被拒。
- `BeamSearchGroup` 的 `top_tokens_buf` 原按 `top_k²` 分配（beam=2048 时达 16MB），
  改为按实际用量分配。

### 9.2 已否决：按 step 裁剪 top_k

直觉上 round≥1 可以把 top-k 收窄到该 step 的最大 degree
（jdsy 词表下 `max_prefix1_degree=415`、`max_prefix2_degree=93`），因为超出 degree 的
top-k 列必然是 padding（logprob `-1e20`）。实测该方案会让服务在首个请求返回后崩溃。

根因在 `BeamSearchGroup`：它的 kernel 把 `top_probs` 的行宽硬编码为 `beam_width`——
按 `request_idx * beam_width * beam_width` 定位每个 request、按 `align_beam_width2`
跨行（`beam_search_group.cpp` 的 `Psum`）。把 round2 的 top-k 裁到 96 后，
kernel 仍按 512 列去读实际只有 96 列的张量，越界踩内存。

而且这不只是索引问题：该 kernel 分块 TopK 后每块只保留 `align_top_k` 个候选，
最终也只能产出 `align_top_k` 个 beam。要选满 `beam_width` 个 beam 就必须
`top_k >= beam_width`。换言之 `top_k == beam_width` 是这个算子的语义契约，
不是可调参数。

要真正吃到裁剪收益，得先重构 `BeamSearchGroup`，把「候选列宽」和「每块选多少」
这两个当前都由 `beam_width` 兼任的角色拆开。这个改动的风险和收益需要单独评估。
