# 4. 参考蓝图政策 + 较新稳定版 API + Context7 核实铁律

- **Status（状态）**: Accepted
- **Date（日期）**: 2026-08-14

## Context（背景）

从零 build 引擎有两种走法：净室（clean-room，完全不看现成实现）vs 参考蓝图（学真实系统的形状再重写）。学习目标是「读代码学真东西的形状」，且要让 infrared 长得像真 vLLM 的骨架（engine/scheduler/block manager/worker）。另外，推理生态迭代极快，凭记忆写 API 极易幻觉。

## Decision（决策）

- **参考蓝图，非净室**：允许 agent 拿 **vLLM v1 架构** + 公开的 **"nano-vLLM"** 当**蓝图**学真实形状；但**重新写带讲解注释的代码，不 copy-paste**。infrared 骨架有意贴近真 vLLM，以最大化「读码学真东西」的价值。
- **较新稳定版**：用**较新的稳定版** API/依赖（PyTorch / Triton / `transformers` / FastAPI）。
- **版本号须实证**（ADR-0006 补强）：**具体版本号必须对真实安装（`pip show`）或 PyPI 核实**，并落到实测跑绿的组合；**Context7 只验 API 形状、不保证某版本号存在**，绝不凭断言写版本号。（R2 违反此条、给了幻觉版本，见 ADR-0006。）
- **Context7 铁律**：任何 API/flag/签名**不确定处，一律先用 Context7 MCP 核实**再写，杜绝幻觉参数。

## Consequences（后果）

**正面：**
- 读 infrared 代码 ≈ 读一个讲解版的真 vLLM 骨架，迁移性强。
- 依赖新、参数真，代码在读者机器上更可能一比一跑通。

**权衡 / 负面：**
- 跟随真架构 → 需偶尔回看 vLLM 设计；跟随新版本 → 需注意 API 漂移（用 Context7 消解）。
- 版权/署名：只学形状、重写实现，注明蓝图来源。

**可逆性：** 高——蓝图与版本是过程约束，可随生态更新调整。
