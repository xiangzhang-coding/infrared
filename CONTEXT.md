# CONTEXT · infrared 双语术语表 (Ubiquitous Language)

> 本文件**只放术语定义**，不含实现细节、决策、配置或预算（那些归 `docs/adr/` 与 `docs/spec/`）。
> 双语：术语给 EN / ZH 名称与简明定义。交叉引用用 →。
> **共享推理词汇**（prefill/decode、KV cache、PagedAttention、MHA/MQA/GQA、TTFT/TPOT、roofline、memory/compute-bound、量化格式…）沿用姊妹项目 [`inference-learning-path/CONTEXT.md`](https://github.com/xiangzhang-coding/inference-learning-path/blob/main/CONTEXT.md)，此处**不重复**，只定义 infrared 特有的工程术语。

---

## 边界 The "from-scratch" boundary

- **Engine / 引擎** — 一个自己做「执行 + 显存管理 + 批处理/调度」的推理运行时。infrared **要造的就是它**。vLLM / ONNX Runtime / TensorRT-LLM / TGI 都是 engine。→ ADR-0003
- **Primitive / 原语** — 我们**站在其上**、但它不替我们做 serving 决策的底层件：PyTorch（GEMM/张量/`torch.cuda` 显存）、HF `transformers`/`safetensors`（仅权重加载 + tokenizer）、Triton（写我们自己的 kernel）、FastAPI（HTTP 壳）。→ ADR-0003
- **Yardstick / 标尺** — 一个现成 engine（vLLM/ONNX Runtime）**仅作性能对照**用，不作 infrared 的执行路径。「达到 vLLM 的 X%」是合法且有价值的一句话。→ ADR-0003

## 引擎内部组件 Engine internals

- **Scheduler / 调度器** — 每一步决定哪些请求进/出当前 batch 的组件；infrared 的心脏。→ Continuous batching
- **Waiting / Running queue（等待 / 运行队列）** — 尚未准入 vs 正在批内解码的请求集合。
- **Block manager / 块管理器** — 把 KV cache 切成定长块、按需分配/回收的分配器（PagedAttention 式）。→ Block table
- **Block table / 块表** — 每个序列「逻辑 KV 位置 → 物理块」的映射，等价于虚拟内存页表。
- **Worker / 工作器** — 实际持有模型权重、在设备上跑 forward 的执行体。
- **Step / 步** — 引擎主循环的一次迭代：调度一批 → forward 一次 → 采样 → 追加 KV → 回收完成的序列。

## 批处理 Batching

- **Static batching / 静态批处理** — 攒一批一起 pad + 生成，全批跑完才收；受最长序列拖累、利用率低。→ 是引出连续批的 baseline。
- **Continuous batching / 连续批处理** — 每步动态准入/退出：序列一完成就腾位给等待请求插入。infrared 的 #1 吞吐杠杆。→ Scheduler
- **Batch fill rate / 批填充率** — 运行槽位被真实占用的比例随时间的曲线；连续批+分页是否奏效的直接证据。→ ADR-0002

## 度量 Metrics（"done" 的定义）

- **Goodput / 有效吞吐** — 满足 SLO（p99 TTFT / p99 TPOT ≤ 阈值）前提下的 req/s。「高并发」的诚实度量，非饱和吞吐。→ Knee
- **Knee / 并发拐点** — 仍满足 SLO 的最高负载点；压测中 request-rate 上扫到 SLO 破裂前的那一点。
- **GPU utilization / GPU 利用率** — achieved occupancy / SM 利用率（torch profiler / nsys 读出）。「高利用率」证据之一。
- **KV-block occupancy / KV 块占用率** — 已分配 KV 块中被真实使用的比例随时间。「高利用率」证据之二。
- **Efficiency / 效率** — 每 GPU-秒的 output tok/s（吞吐/成本），及 Triton kernel 相对朴素 gather 的加速比。
- **Before→after ladder / 优化前后阶梯表** — `静态批 → 连续批 → +paged KV → +Triton kernel` 逐档给出 (correctness, throughput, goodput, utilization) 的表；每档是**自己造的机制**而非一个 flag。→ ADR-0002

## 阶梯 The tiers (T0–T6)

- **T0–T6** — infrared 的建造阶梯，从「单请求正确生成」到「投机解码/量化/multi-LoRA/张量并行」。每档 = 一个亲手实现的机制 + 一次对上一档的度量。定义见 `README.md` 与 `docs/spec/0001`。

## 正确性 Correctness

- **Correctness oracle / 正确性对拍标尺** — 用 HF `transformers` 加载**同一权重**、greedy + 固定 seed 跑参考输出，比对 infrared 的 logits / 输出确认 forward 没写错。既是权重来源、又是正确性 gate。→ ADR-0005
