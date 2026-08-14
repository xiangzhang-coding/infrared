# Spec · infrared —— 从零 LLM 推理引擎 (from-scratch inference engine)

> 本 spec 综合本次 grilling 共识、`CONTEXT.md`（术语表）与 `docs/adr/0001–0005` 而成。
> 术语一律沿用 `CONTEXT.md`；所有决策受 `docs/adr/` 约束。
> **执行进地图**（见 Further Notes）：本项目由 agent 按票实现，作者读代码学习——故本 spec 之后**会**产出引擎代码（与姊妹站 spec 不同）。

---

## Problem Statement（问题陈述）

我已有 [`inference-learning-path`](https://github.com/xiangzhang-coding/inference-learning-path)——一条教我**读懂 + 会调 vLLM**、并冲刺大厂推理 Infra 面试的系统学习路径。但它 (ADR-0002) 刻意把深度边界钉在「读源码 + 调旋钮 + 应用层」，把**从零造引擎、手写 kernel** 归入 backlog。要真正获得「优化推理服务/框架」的深层知识，我需要一个**动手项目**：把「高并发 / 高利用率 / 高效」从**我去调的旋钮**，变成**我亲手实现的机制**。缺的是一个完整、端到端、可运行、可度量的从零推理引擎。

## Solution（解决方案）

一个用 **Python + PyTorch** 从零写的 **LLM 推理引擎**，按 **T0–T6 阶梯**循序渐进建造：

- **T0** 单请求正确生成（自实现 Qwen2.5 forward + 每请求 KV + 采样）→ **T1** 静态批 + 请求队列 + HTTP → **T2** 连续批调度器 → **T3** paged KV 块管理 → **T4** 效率档（Triton paged-attn kernel / prefix caching / chunked prefill / CUDA graphs）→ **T5** 薄服务层（流式 / metrics / 压测 harness）→ **T6** 之外（投机解码 / 量化 / multi-LoRA / 张量并行）。

每档 = **一个亲手实现的机制 + 一次对上一档的度量**。度量脊柱（correctness / throughput / goodput / utilization）在每档复跑，产出一张 `静态批 → 连续批 → +paged → +Triton` 的**优化前后阶梯表**——每档是自己造的机制而非一个 flag。这是最硬的作品与面试答案。

**边界**（ADR-0003）：造 serving 机制，站在 PyTorch / HF 权重加载 / Triton / FastAPI 之上，**绝不**拿现成 engine（vLLM/ONNX Runtime/TensorRT-LLM/TGI）当执行路径（它们只作**标尺**）。

## Goals / 学习目标（build 目标，而非用户故事）

1. 亲手实现 Qwen2.5 dense forward（RMSNorm/RoPE/GQA/SwiGLU），并用 HF 对拍验证正确。
2. 理解 KV cache 如何增长、如何成为吞吐/显存上限——因为我自己管理它。
3. 亲手把静态批演进到连续批，**量出**利用率与吞吐的跃升。
4. 亲手实现 PagedAttention 式块管理器 + 块表，**量出**显存利用率与并发数的提升。
5. 亲手写一个 Triton paged-attention kernel，量出相对朴素 gather 的加速。
6. 建一个度量脊柱（goodput@SLO / knee / GPU util / KV 占用 / batch 填充），能诚实地画出并发拐点。
7. 包一个薄服务层（OpenAI 兼容 + 流式 + metrics + 压测 harness），让引擎端到端可用、可测。
8. 逐步接入进阶机制（投机解码 / 量化 / multi-LoRA / TP），把学习站里「读过」的东西亲手「写」一遍。

## Implementation Decisions（实现决策）

- **定位与深度**：遵循 **ADR-0001**——build 引擎、刻意跨过学习站 ADR-0002 的 build 边界；仍不手写 CUDA C++（用 Triton）。
- **"done" 度量**：遵循 **ADR-0002**——goodput@SLO / knee / GPU util / KV 块占用 / batch 填充 / 优化前后阶梯表。
- **from-scratch 边界**：遵循 **ADR-0003**——build serving 机制、stand on 原语、never engine-as-execution（engine 仅作标尺）。
- **参考蓝图与版本**：遵循 **ADR-0004**——nano-vLLM + vLLM v1 架构作蓝图，重写带讲解注释、不 copy-paste；用较新稳定版 API/依赖，不确定处一律 Context7 核实。
- **硬件/模型基线**：遵循 **ADR-0005**——沿用学习站 ADR-0001（单 4090 / ¥500 / 无卡模式）；实现 Qwen2.5 架构；正确性开发用 `Qwen2.5-0.5B-Instruct`、北极星压测用 `Qwen2.5-7B`；HF 既是权重来源又是正确性对拍标尺；TP 验证临时租 2 卡机短开即关。
- **栈**：Python + PyTorch（引擎/数学）；Triton（自写 kernel）；HF `transformers`/`safetensors`（仅加载权重 + tokenizer）；FastAPI/uvicorn（HTTP 壳）。

## Testing Decisions（测试决策）

- **什么是好测试**：只验证**外部可观察行为**，不测实现细节。
- **Seam A（正确性闸门）**：infrared 在固定输入 + greedy + seed 下的 logits/输出**对拍 HF `transformers`** 参考实现（同一权重）——这是引擎「没写错」的最高 seam。每个改动机制的档都必须过。
- **Seam B（度量脊柱可跑）**：度量 harness 能对任一配置产出 (correctness, throughput, goodput, utilization) 一行，并驱动阶梯表。**相对档收益**可复现，而非追绝对 SOTA 数字。
- **质量 gate**：任何机制档若破坏正确性对拍（或坍塌某类别）→ 回退并写下原因（对齐学习站 Capstone 的 revert 纪律）。
- **Prior art**：全新仓库，暂无既有测试。

## Out of Scope（不在范围内）

- **手写 / 优化 CUDA C++**——继承学习站 ADR-0002，进 backlog；infrared 只写 Triton。
- **与 vLLM 拼绝对性能**——只比**相对档收益**与「%-of-vLLM」标尺，不追 SOTA。
- **训练 / 微调 infra**——聚焦推理。
- **多模态 / embedding / reranker serving**——backlog。
- **真实生产集群部署（K8s / 自动扩缩 fleet / 公网 SLA）**——T5 只是**演示级**服务壳，不做生产 fleet；后续 effort。
- **(D) 直接改/优化真 vLLM**——是 infrared 之后的下一个 effort，不在本图。

## Further Notes（补充说明）

- **执行进地图 (override)**：本 wayfinder 图不止规划，还长出**实现票**给 agent 建；作者读代码学习。地图 `Notes` 记此 override。
- **tracker 房规**（对齐姊妹站）：GitHub issues；一个长期开着的 Spec/Map issue 作锚；实现票挂 `ready-for-agent`；阻塞用正文 `Blocked by: #N` 约定；wayfinder 图挂 `wayfinder:map`。
- **主机铁律**：所有 gh/git 操作 **`GH_HOST=github.com`**（本机另挂 SAP 企业版 host，绝不误建）。
- **脱敏**：仓库不含任何雇主/公司内部信息；Git 提交身份用 GitHub noreply 邮箱。
- **仓库**：`github.com/xiangzhang-coding/infrared`（public）。
