# 3. from-scratch 边界：造 serving 机制，站在原语上，绝不拿 engine 当执行路径

- **Status（状态）**: Accepted
- **Date（日期）**: 2026-08-14

## Context（背景）

形状 (A) 的全部意义在于**亲手造 serving 机制**。若无一条清晰的「什么自己造、什么站在现成件上」的边界，很容易滑向「调用现成 engine」——那样就没有调度器/KV 管理器/paged cache 留给我造，学习目标被掏空。（grilling 中「能不能用 ONNX」正是此边界的触发点。）

## Decision（决策）

| | 东西 | 为什么 |
|---|---|---|
| **✅ 自己造**（学习面） | forward 接线 + attention/KV 路径、KV cache（连续→paged 块管理）、调度器（静态→连续批）、请求队列+server、采样循环、Triton kernel、prefix caching、chunked prefill、CUDA graphs、TP、量化接入 | 这些**就是**「推理服务优化」本身 |
| **🟩 站在上面**（原语/管线） | **PyTorch**（GEMM/张量/`torch.cuda` 显存）、**HF `transformers`/`safetensors`**（**仅**权重加载 + tokenizer，**绝不用 `.generate()`**）、**Triton**（写自己的 kernel）、**FastAPI/uvicorn**（HTTP 壳） | 是原语，不替我做 serving 决策 |
| **⛔ 不作引擎，但可作标尺** | vLLM / ONNX Runtime / TensorRT-LLM / TGI / llama.cpp | 不能当 infrared 执行路径；**可当 benchmark 标尺**（「达到 vLLM 的 X%」） |

**底不再往下**：不裸写 CUDA C++、不脱离 PyTorch（PyTorch 给快 GEMM，让我专注 serving 机制）。唯一写 kernel 处 = Triton paged-attention。

## Consequences（后果）

**正面：**
- 学习面（自己造的部分）最大化，且不被 GEMM/kernel 细节淹没。
- 「engine vs primitive vs yardstick」成为清晰的判断准则，防止范围滑移。

**权衡 / 负面：**
- 引擎绝对性能受 PyTorch eager + 无手写 CUDA 限制——刻意取舍（ADR-0001/0002）。

**可逆性：** 中——个别原语可替换（如换 attention 后端），但「绝不拿 engine 当执行路径」是不可动摇的项目定义，故立此 ADR。
