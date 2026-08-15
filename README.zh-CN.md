# infrared

[English](README.md) · **中文**

> 一个**从零手写的 LLM 推理引擎** —— 目标是通过*亲手实现*各项机制（连续批处理、分页 KV cache、调度器、Triton kernel）来学习推理服务优化，而不是去调别人引擎的开关。外加一层薄薄的 serving 层，让它能端到端跑起来。

`infrared` 是 [`inference-learning-path`](https://github.com/xiangzhang-coding/inference-learning-path)（一个教你*读懂并调优* vLLM 的双语教程站）的动手姊妹项目。那个站点刻意止步于「读源码、调旋钮」，而 `infrared` 越过了这条线，**把引擎造出来** —— 在这里，高并发 / 高利用率不再是你去拧的旋钮，而是你自己拥有的机器。

## 建造阶梯（T0 → T6）

一档一档地造，每一档都是你亲手实现的一个机制，然后**度量**它相对下一档的收益：

| 阶梯 | 你要造的东西 | 学到什么 |
|---|---|---|
| **T0** | 单请求正确生成 —— Qwen2.5 forward（RMSNorm/RoPE/GQA/SwiGLU）、单请求 KV、采样循环 | prefill/decode、KV 增长 |
| **T1** | 静态批处理 + 请求队列 + HTTP 服务（OpenAI 风格） | 批处理、padding 浪费、静态批为何利用率低 |
| **T2** | **连续批处理**调度器 | 吞吐的头号杠杆；调度器即引擎的心脏 |
| **T3** | **分页 KV cache**（PagedAttention 式块管理器） | vLLM 的核心创新；显存利用率 |
| **T4** | 效率：Triton 分页注意力 kernel、前缀缓存、分块预填充、CUDA 图 | 性能的那一半 |
| **T5** | serving 层：响应流式输出、实时指标端点、serving/ops 加固（离线指标 + 压测**脊柱**已提前落地 —— issue #7） | 运维 / serving 系统的那一半 |
| **T6** | 进阶：投机解码、量化、multi-LoRA、张量并行 | 高阶 serving |

## 「done」度量什么

- **高并发** = **有效吞吐（goodput）**：满足 SLO（p99 TTFT / TPOT）前提下的 req/s，以及并发**拐点（knee）**。
- **高利用率** = GPU 计算利用率 + **KV 块占用率 / 批填充率**随时间的曲线 —— 证明调度器与分页确实在起作用。
- **高效率** = 每 GPU-秒的 output tok/s，以及 Triton kernel 相对朴素 gather 的加速比。
- 决定性的产物：一张 **`静态批 → 连续批 → +分页 → +Triton`** 的优化前后对照表，其中每一档都是*你亲手造的机制*。

## 「从零手写」的边界

**要造**的是 serving 机器本身（forward 接线 + 注意力/KV 路径、KV cache、调度器、批处理、服务器、Triton kernel、前缀缓存……）。**可以站在**原语之上 —— PyTorch（GEMM/张量）、HF `transformers`/`safetensors`（仅权重加载 + tokenizer，绝不用 `.generate()`）、Triton（用来写*我们自己的* kernel）、FastAPI（HTTP 壳）。**绝不**把某个现成的推理*引擎*（vLLM / ONNX Runtime / TensorRT-LLM / TGI）当作执行路径 —— 那会把整件事的意义掏空；它们只允许作为**性能对照标尺**存在。

## 基准硬件

单卡 RTX 4090（24 GB），经由 AutoDL，预算 ¥500（沿用自 `inference-learning-path` 的 ADR-0001）。正确性开发用 `Qwen2.5-0.5B-Instruct`（对 CPU/Mac 友好）；标题级基准测试跑 `Qwen2.5-7B`。

## 仓库结构

T0（单请求 forward）、T1（静态批 + HTTP）、**T2（连续批）、T3（分页 KV），以及完整的 T4 效率档（前缀缓存、分块预填充、Triton 分页注意力 kernel、CUDA 图 decode）**均已实现并验证（HF 对拍 + 优化前后阶梯表都跑绿；Triton kernel 与 CUDA 图都带有数值等价的 CPU 回退路径，其 GPU 对拍 + 加速比留待 AutoDL 上的 4090 运行去验证），另加**度量脊柱**（issue #7），可度量任意配置；T5/T6 尚未开工。命名沿用 R1 架构蓝图（`docs/research/vllm-v1-nano-vllm-blueprint.md`），并沿最关键的那道接缝切分 —— 请求编排 vs 模型执行：

```
infrared/
  config.py            # EngineConfig（block_size、预算……）—— 纯数据
  model/               # 模型执行（batch-first；B=1 即 T0 路径）
    config.py          #   Qwen2Config，从 HF config.json 读取                (T0)
    layers.py          #   RMSNorm / RoPE / GQA attention / SwiGLU           (T0)
    triton_attention.py #  融合分页注意力 Triton kernel + 朴素回退           (T4c)
    inputs.py          #   positions + 加性因果/padding mask                 (T1)
    qwen2.py           #   Qwen2.5 forward + safetensors 加载器（tied lm_head）(T0)
    sampler.py         #   greedy / temperature                             (T0)
    generate.py        #   单请求 prefill→decode + tokenizer 路径           (T0)
    model_runner.py    #   Worker 接缝（stub）                              (T3+)
  engine/              # 请求编排
    static_batch.py    #   静态批：左 pad prefill + 齐步 decode             (T1)
    engine.py          #   Static + ContinuousBatch 引擎（busy loop）       (T1/T2)
    paged_engine.py    #   PagedBatchEngine：分页 KV + 批量 decode          (T3)
    cuda_graph.py      #   CUDA 图 decode capture/replay（仅 GPU）          (T4d)
    scheduler.py       #   连续批调度器（waiting/running）                   (T2)
    sequence.py        #   Sequence 状态机（+ block_table）                 (T2/T3)
  cache/
    kv_cache.py        #   连续的单请求 KV cache（batch-first）             (T0/T1)
    block_manager.py   #   PagedAttention 块池 + 块表                        (T3)
    paged_kv_cache.py  #   共享分页 K/V 池 + scatter/gather 接缝            (T3)
  server/app.py        #   FastAPI OpenAI /v1/completions（非流式）         (T1)
  bench/               # 度量脊柱 —— 「done」阶梯表（驱动每一档）
    metrics.py         #   percentiles / TTFT·TPOT / goodput / knee（纯函数）(脊柱)
    workload.py        #   请求类别 + 泊松到达过程（纯函数）                 (脊柱)
    report.py          #   Markdown/CSV 阶梯表 + knee 曲线渲染器（纯函数）    (脊柱)
    harness.py         #   压测驱动 + correctness/throughput/util + measure  (脊柱)
    __main__.py        #   `python -m infrared.bench` CLI                   (脊柱)
```

## 开发

运行时依赖（torch / triton / transformers / …）钉在 `docs/research/deps-and-qwen25-arch.md` 列出的版本上，且是 **Linux + GPU** 的（triton 只发 Linux wheel）。在无 GPU 的机器上走 dev 路径 —— 它跳过这些依赖，但仍能让测试与 linter 跑绿：

```bash
make install-dev   # 可编辑安装 + pytest/ruff，不装 torch/triton
make test          # 单元 + 冒烟测试（对拍测试在模型未缓存时跳过）
make lint          # ruff check（经由 uvx）
make install       # 完整运行时安装 —— 仅 Linux + GPU
```

脚手架级别的测试从不碰 GPU、也不下载模型，因此 `make test` / `make lint` 在任何平台都能过。

### 正确性 gate（Seam A）

T0 的验收标准是：infrared 的 forward 在同一份权重上与 HF `transformers` 一致（greedy 输出 + 第一步 logits）。这道 gate 位于 `tests/test_parity.py`，在本地未缓存 `Qwen2.5-0.5B-Instruct` 时会**跳过**。要运行它（一次性下载约 1 GB，在 CPU 上跑）：

```bash
make parity        # 拉取 0.5B 权重，然后跑 HF 对拍测试
```

HF 只被用作权重来源（+ tokenizer）与参考对拍标尺 —— 绝不经由 `.generate()`（ADR-0003）。

### Serving（T1）

启动 OpenAI 兼容服务器（默认加载 `Qwen2.5-0.5B-Instruct`；用 `INFRARED_MODEL` 覆盖）：

```bash
uvicorn infrared.server.app:build_app --factory --port 8000
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

静态批处理刻意是 T2 的朴素 baseline：响应里带一个非标准的 `infrared_batch` 字段（`prompt_pad_tokens`、`decode_slack_tokens`……），让你能*亲眼看见* padding 浪费与队头阻塞 —— 也正是连续批调度器将要消除的东西。

### 度量脊柱（「done」阶梯表）

一条命令就能把任意引擎配置化成一行诚实的数据 —— `(correctness, throughput, goodput, utilization)` —— 以及它背后的 request-rate **knee 扫描**（ADR-0002，`CONTEXT.md` §Metrics）。它是未来每一档都会接入的统一度量入口。

```bash
make bench                                  # CPU 上的微型随机模型（无 GPU、无下载）
python -m infrared.bench --model Qwen/Qwen2.5-0.5B-Instruct   # 真实权重（已缓存）
python -m infrared.bench --rates 5,25,100,400 --ttft-ms 500 --tpot-ms 100
```

每一列的含义，以及它们如何保持诚实：

- **correctness** —— 对每个 prompt（greedy）与 oracle 做 A/B，*按类别*统计；要通过，需每个类别都逐字一致。默认 oracle 是 T0 单请求路径，而它本身受 HF 对拍 gate 约束，因此一次匹配就传递性地等价于「相对 HF 的 batch 不变性」。
- **throughput** —— 在固定的 decode 密集型 shape 上的持续 output tok/s。
- **goodput / knee** —— 以开环方式按递增速率投放请求（泊松到达）；**goodput** 是单独满足 SLO 的请求速率，**knee** 是 p99 TTFT/TPOT 仍能满足的最高投放速率 —— 而*不是*饱和吞吐。
- **utilization** —— T1 的证据是混合负载上的**批填充率**（真实 decode 工作量 ÷ 被 pad 的齐步网格），静态批的 padding + 队头浪费会在这里暴露出来。GPU 利用率在 CUDA 上补齐；KV 块占用率随分页 KV（T3）到位。

纯数学部分（percentiles、TTFT/TPOT、goodput、knee、渲染器）位于 `bench/metrics.py`、`bench/workload.py`、`bench/report.py`，无需 torch 即可单元测试；`bench/harness.py` 是那层薄驱动，从运行中的引擎记录真实 trace。**方法与负载可复现**（带 seed 的泊松到达 + 带 seed 的 prompt）；绝对墙钟延迟依机器而异，所以这份产物讲的是*相对*的逐档收益，而非绝对 SOTA（spec §Testing，Seam B）。

## 状态

🛠️ **建造中** —— T0（单请求 forward，受 HF 对拍 gate 约束）、T1（静态批 + OpenAI 兼容 HTTP）、T2（连续批）、**T3（分页 KV cache）**，以及 **T4 效率档**（前缀缓存、分块预填充，及一支手写的 **Triton 分页注意力 kernel**）均已就位，另加可为任意配置打分、并叠出 `静态批 → 连续批 → +分页 → +Triton` 优化前后阶梯表的**度量脊柱**（`python -m infrared.bench`）。T2 的迭代级调度器把批填充率推到 100% 并暴露出每请求的真实 TTFT（在每条序列产出首 token 的瞬间打上时间戳 —— HTTP 层仍以非流式方式一次性返回整段补全）；T3 按需从共享池中取定长 KV 块（无最坏情况预留、无碎片 → 更高的 KV 块占用率、每单位 KV 预算能容纳更多并发序列），并**把 decode 步批量化**跨整个运行集 —— 在池压力下带重算式抢占，重新拿回吞吐/goodput 杠杆。T4 跨请求复用共享的 prompt 前缀 KV，把长 prefill 与 decode 交织，并 —— 在 CUDA 上 —— 把分页 gather + scaled-dot-product + online-softmax 融合进一支 Triton kernel，且把 decode 步作为捕获的 CUDA 图重放（CPU 回退到朴素 eager 路径；GPU 对拍 + 加速比留给 AutoDL 上的 4090 运行）。接下来是 T5/T6。整个计划以一个 [wayfinder 地图 issue](https://github.com/xiangzhang-coding/infrared/issues) 的形式存在，附带建造工单。settled 的决策与术语表见 `docs/spec/`、`docs/adr/` 与 `CONTEXT.md`。

## 许可证

Apache-2.0 —— 见 [LICENSE](LICENSE)。
