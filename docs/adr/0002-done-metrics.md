# 2. "done" 的度量：goodput@SLO / knee / 利用率证据 / 优化前后阶梯表

- **Status（状态）**: Accepted
- **Date（日期）**: 2026-08-14

## Context（背景）

项目北极星是「高并发 / 高利用率 / 高效」——但这三个词模糊，不钉成可测指标就无法判断任何一档机制是否奏效，也无法产出可辩护的作品。需要一套**贯穿全阶梯、每档复跑**的度量定义。

## Decision（决策）

把三个模糊词钉成以下可测指标（术语入 `CONTEXT.md`）：

- **高并发 = goodput**：满足 SLO（p99 TTFT / p99 TPOT ≤ 阈值）前提下的 **req/s**，及**并发拐点 knee**（request-rate 上扫到 SLO 破裂前的最高负载）。**不用饱和吞吐充数。**
- **高利用率 = 两个证据**：**(a) GPU 计算利用率**（torch profiler / nsys 的 achieved occupancy / SM util）+ **(b) KV 块占用率 / batch 填充率随时间曲线**——用来**证明**连续批 + 分页真在工作。
- **高效 = 每 GPU-秒的 output tok/s**（吞吐/成本），外加 **Triton kernel 相对朴素 gather 的加速比**。
- **杀手级产物 = 优化前后阶梯表**：`静态批 → 连续批 → +paged KV → +Triton kernel`，逐档给出 (correctness, throughput, goodput, utilization)。**每档改一个自己造的机制**（对齐学习站 Capstone「一次一变、可归因、质量 gate、revert 要写下来」的纪律）。

## Consequences（后果）

**正面：**
- 每档机制都有「奏效证据」，而非「我把它做快了」的 vibe。
- 度量脊柱统一 → 各档 delta 可归因、可复现（相对收益）。

**权衡 / 负面：**
- 需先建度量 harness（一张早期票），否则前几档无处度量。
- 若把 v1 射程砍到 T2，(b) 里「块占用率」需换成别的利用率证据。

**可逆性：** 中——阈值/SLO 是配置，可调；但「用 goodput 而非饱和吞吐、每档配利用率证据」这一**度量姿态**是全项目基线，故立此 ADR。
