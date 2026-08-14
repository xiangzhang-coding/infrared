# 6. R2 的依赖版本号系幻觉，已更正为实测真实版本

- **Status（状态）**: Accepted
- **Date（日期）**: 2026-08-14
- **更正 / 补强**: R2（issue #3）在 `docs/research/deps-and-qwen25-arch.md` §1/§4 给出的依赖版本；补强 ADR-0004。

## Context（背景）

研究票 R2（#3）声称「版本号权威来源是 PyPI `info.version`」，据此在研究文档与 `pyproject.toml` 钉了：`torch 2.13.0` / `triton 3.7.1` / `transformers 5.15.0` / `safetensors 0.8.0` / `uvicorn 0.52.3`（`fastapi 0.141.1`）。

真正动手时发现：**这些版本号多为幻觉 / 不存在的未来版**，`uv pip install -e .` 在真实环境直接失败。即 R2 谎称查过 PyPI，却给出了未经核实的版本号——而 ADR-0004 的 Context7 铁律只覆盖 **API 形状**，没能拦住**版本号**幻觉。

本仓 **70 项测试 + HF parity 实测跑绿**的真实版本是：`torch 2.12.0` / `transformers 4.56.2` / `safetensors 0.7.0` / `fastapi 0.141.1`（R2 此项恰好属实）/ `uvicorn 0.45.0`；`triton` 仅 Linux wheel、由 torch 传递解析。

## Decision（决策）

1. **依赖钉到实测跑绿的真实版本**（上列）；`pyproject.toml` 与研究文档 §1/§4 据此更正。
2. **`triton` 不自钉**——交给 torch 的 Linux CUDA wheel 传递解析；待 T4 在 GPU 机上观测到确切版本再显式钉。
3. **补强 ADR-0004**：**版本号必须对真实安装（`pip show`）/ PyPI 核实**并落到实测跑绿组合；Context7 只验 API 形状、不保证某版本号存在。
4. 研究文档 **§2/§3 的 Qwen2.5 架构超参与权重命名保持不变**——它们对真实 `config.json` / `.safetensors` 核过，且 **T0 的 HF parity 已反证其正确**。

## Consequences（后果）

**正面：** 仓库可真正安装、可复现；决策记录诚实留痕，不掩盖 R2 的幻觉。
**权衡 / 负面：** 「较新稳定版」让位于「实测跑绿的真实版本」——可能不是 PyPI 最新，但**可复现优先**。
**可逆性：** 高——版本是配置项，日后在 GPU 机实测后可上调，并同步更新此 ADR 与研究文档 §1。
