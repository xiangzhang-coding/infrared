# 1. 定位与深度：build 引擎，刻意跨过学习站的 build 边界（但仍不手写 CUDA C++）

- **Status（状态）**: Accepted
- **Date（日期）**: 2026-08-14

## Context（背景）

姊妹项目 [`inference-learning-path`](https://github.com/xiangzhang-coding/inference-learning-path) 教「读懂 + 会调 vLLM」，其 **ADR-0002** 把学习深度边界钉在 **A+C**（会推理原理、会调优、能读源码、会写几个 Triton kernel），并把**从零造引擎、手写 CUDA C++** 明确归入 backlog。要真正获得「优化推理服务/框架」的深层知识，需要一个动手项目把「高并发/高利用率」从**旋钮**变成**亲手实现的机制**。

## Decision（决策）

`infrared` = **从零 build 一个推理引擎**，**刻意跨过**学习站 ADR-0002 的「不 build」边界。但**仍不手写 CUDA C++**（继承学习站 ADR-0002 的这一半）——唯一写 kernel 的地方是 **Triton**。

- 形状：**(A) 从零最小引擎 + (C) 薄服务层 = 端到端**；(D) 直接改真 vLLM 留作后续 effort。
- 与姊妹站的关系：**companion（互补），非重复**。infrared 复用学习站的**共享推理术语**（`CONTEXT.md` 交叉引用）、**eval/度量心智**与 **Capstone 的 revert 纪律**；学习站教「读」的东西（continuous batching、PagedAttention、Triton kernel），infrared 亲手「写」。

## Consequences（后果）

**正面：**
- 补齐学习站故意留白的「build」那一半，学习闭环从「读+调」延伸到「造+量」。
- 最硬的面试作品：「我写过一个连续批处理调度器 / paged KV 管理器」。

**权衡 / 负面：**
- 工程量远大于写文档；靠 wayfinder 分期 + 迷雾管理，避免一次面对全部 T0–T6。
- 不手写 CUDA C++ → 引擎绝对性能不与 vLLM 同台；这是刻意取舍（学的是**引擎/服务设计**，非 kernel 竞赛），只比相对档收益 + %-of-vLLM 标尺（ADR-0002/0003）。

**可逆性：** 低——整个仓库的存在与结构都建立在此定位上，故立此 ADR 作根决策。
