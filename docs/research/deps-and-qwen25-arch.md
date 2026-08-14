# Research: 较新稳定版依赖 + Qwen2.5 架构超参

> AFK research note for infrared. 解决 issue #3（R2）。所有版本号可回溯到具体来源；架构超参直接取自 Hugging Face 上两个模型的一手 `config.json` 与实际 `.safetensors` 文件头。
>
> - **调研日期**: 2026-08-14
> - **约束来源**: ADR-0004（较新稳定版 + Context7 核实铁律）、ADR-0005（Qwen2.5 dense，0.5B 开发 / 7B 压测）

> ⚠️ **版本更正（2026-08-14 · ADR-0006）**：本文档 **§1 与 §4** 最初由 R2 给出的依赖版本号**系幻觉、未真正核实**——`torch 2.13.0` / `transformers 5.15.0` / `safetensors 0.8.0` / `uvicorn 0.52.3` / `triton 3.7.1` **均不存在或未验证**，`pip install` 直接失败。已更正为**本仓实测通过**（70 测试 + HF parity 全绿）的真实版本：`torch 2.12.0` / `transformers 4.56.2` / `safetensors 0.7.0` / `fastapi 0.141.1`（此项 R2 当初恰好属实）/ `uvicorn 0.45.0`；`triton` 不由我们自钉，交给 torch 的 Linux wheel 传递解析。**§2 / §3 的 Qwen2.5 架构超参与权重命名是对真实 `config.json` / `.safetensors` 核过的，属实、且已被 T0 的 HF parity 反证正确，不受此更正影响。** 详见 ADR-0006。

---

## 1. 依赖版本表（实测通过的真实版本）

> ⚠️ 本节已按 ADR-0006 更正。原表由 R2 断言「取自 PyPI `info.version`」，实为**未核实的幻觉版本**。下表钉的是**本仓 70 测试 + HF parity 实际跑绿**的真实安装版本（`pip show` 印证）——这才是可复现的权威来源。

| 依赖 | 钉住的真实版本 | 来源 / 核实 | 备注 |
|---|---|---|---|
| `torch` | **2.12.0** | 本仓实测环境 `pip show`（测试全绿） | Linux CUDA wheel 自带并钉住其配对 triton |
| `triton` | *不自钉* | 由 `torch` 的 Linux wheel 传递解析 | 仅 Linux wheel；T4 才 `import triton`，届时在 GPU 机上观测到确切版本再钉 |
| `transformers` | **4.56.2** | 本仓实测环境 `pip show` | **4.x（不是 R2 说的 5.x）**；用法见 §4 |
| `safetensors` | **0.7.0** | 本仓实测环境 `pip show` | 加载 API：`safe_open` / `load_file`，见 §3.3 |
| `fastapi` | **0.141.1** | 本仓实测环境 `pip show` | 服务层用；R2 当初此项恰好属实 |
| `uvicorn` | **0.45.0** | 本仓实测环境 `pip show` | ASGI server，配 FastAPI |

> 教训（ADR-0006）：**版本号必须对真实安装 / PyPI 核实**，Context7 只验 **API 形状**、不保证某版本号存在。R2 谎称查过 PyPI 却给了未来/不存在的号，直到真装才暴露。

---

## 2. Qwen2.5 架构超参对照表（0.5B-Instruct vs 7B-Instruct）

一手来源：两个模型的 `config.json`
- `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json`
- `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/config.json`

两者 `model_type` 均为 **`qwen2`**，`architectures` 均为 **`Qwen2ForCausalLM`**（Qwen2.5 沿用 Qwen2 建模代码）。

| 超参 (config key) | Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct | 备注 |
|---|---|---|---|
| 层数 `num_hidden_layers` | **24** | **28** | |
| hidden size `hidden_size` | **896** | **3584** | |
| attention heads `num_attention_heads` | **14** | **28** | |
| **KV heads `num_key_value_heads` (GQA)** | **2** | **4** | GQA group = Q/KV = **7** 两档都一样 |
| head_dim `hidden_size / num_attention_heads` | **64** (896/14) | **128** (3584/28) | config 无显式 `head_dim`，按此推导；由 `.safetensors` 形状印证（见 §3.1） |
| intermediate size `intermediate_size` | **4864** | **18944** | SwiGLU MLP 中间维 |
| RoPE theta `rope_theta` | **1000000.0** | **1000000.0** | 两档相同 |
| vocab `vocab_size` | **151936** | **152064** | **不同**：7B 更大（词表 padding 差异），embed/lm_head 行数随之不同 |
| RMSNorm eps `rms_norm_eps` | **1e-06** | **1e-06** | |
| tie embeddings `tie_word_embeddings` | **true** | **false** | **关键差异**：0.5B 共享 embed↔lm_head（无独立 `lm_head.weight`）；7B 独立 |
| `hidden_act` | silu | silu | 即 SwiGLU 的门控激活 |
| `max_position_embeddings` | 32768 | 32768 | `use_sliding_window=false`，推理时不启用滑窗 |
| `bos_token_id` / `eos_token_id` | 151643 / 151645 | 151643 / 151645 | chat 停止符见 `generation_config.json`：`eos_token_id=[151645, 151643]` |
| `torch_dtype`（权重原始精度） | bfloat16 | bfloat16 | 7B 权重总量 `15,231,233,024` B ≈ 15.2 GB（对齐 ADR-0005 的 “~15GB”） |

派生 KV 投影维度（自实现 forward 要用）：
- **0.5B**：q_proj out = 14×64 = **896**；k/v_proj out = 2×64 = **128**。
- **7B**：q_proj out = 28×128 = **3584**；k/v_proj out = 4×128 = **512**。

---

## 3. 权重命名 / 加载要点（safetensors → 我们的模块映射）

### 3.1 文件布局（HF `/api/models/...` 的 siblings 列表印证）

- **0.5B**：单文件 **`model.safetensors`**（无 shard index；`model.safetensors.index.json` 返回 *Entry not found*）。文件头共 **290** 个张量。
- **7B**：**分片 4 文件** `model-0000{1..4}-of-00004.safetensors` + **`model.safetensors.index.json`**（`weight_map` 共 **339** key，`metadata.total_size = 15231233024`）。加载时先读 index 的 `weight_map`：key → 分片文件名。

### 3.2 权重 key 模式 → infrared 模块映射

下列 key 名与张量形状取自**实际 `.safetensors` 文件头**（0.5B 逐张量读取；7B 由 index `weight_map` 印证），非二手推断。形状以 0.5B 为例（`H=896, I=4864, Hkv=128, V=151936`）。

| safetensors key 模式 | 形状 (0.5B) | 有无 bias | → infrared 模块 |
|---|---|---|---|
| `model.embed_tokens.weight` | `[V, H]` = `[151936, 896]` | — | token embedding（0.5B 同时充当 lm_head，见 tie） |
| `model.layers.{i}.input_layernorm.weight` | `[H]` | — | 注意力前 RMSNorm（pre-norm） |
| `model.layers.{i}.self_attn.q_proj.weight` | `[H, H]` = `[896, 896]` | **有** `q_proj.bias` `[896]` | 注意力 Q 投影 |
| `model.layers.{i}.self_attn.k_proj.weight` | `[Hkv, H]` = `[128, 896]` | **有** `k_proj.bias` `[128]` | 注意力 K 投影（GQA，Hkv<H） |
| `model.layers.{i}.self_attn.v_proj.weight` | `[Hkv, H]` = `[128, 896]` | **有** `v_proj.bias` `[128]` | 注意力 V 投影（GQA） |
| `model.layers.{i}.self_attn.o_proj.weight` | `[H, H]` = `[896, 896]` | **无** | 注意力输出投影 |
| `model.layers.{i}.post_attention_layernorm.weight` | `[H]` | — | MLP 前 RMSNorm |
| `model.layers.{i}.mlp.gate_proj.weight` | `[I, H]` = `[4864, 896]` | **无** | SwiGLU 门控投影 |
| `model.layers.{i}.mlp.up_proj.weight` | `[I, H]` = `[4864, 896]` | **无** | SwiGLU 上投影 |
| `model.layers.{i}.mlp.down_proj.weight` | `[H, I]` = `[896, 4864]` | **无** | SwiGLU 下投影 |
| `model.norm.weight` | `[H]` | — | 最终 RMSNorm（lm_head 前） |
| `lm_head.weight` | `[V, H]` | — | **仅 7B 存在**；0.5B `tie_word_embeddings=true`，无此 key，须复用 `model.embed_tokens.weight` |

**Qwen2 关键坑（自实现 forward 必须对齐）**：
1. **QKV 有 bias、O 与 MLP 无 bias**——这是 Qwen2/2.5 区别于 Llama 的显著特征；漏掉 q/k/v 的 bias 会静默拉低正确性。
2. **权重是 `nn.Linear` 约定的 `[out, in]`**，forward 里 `x @ W.T`（或 `F.linear(x, W, b)`）。
3. **0.5B 的 lm_head 权重绑定**：加载时若字典里没有 `lm_head.weight`，直接把 `model.embed_tokens.weight` 当 lm_head 用（tied）。
4. Linear 层数总数：每层 7 个 weight（q/k/v/o + gate/up/down）+ 2 个 norm + 3 个 q/k/v bias；0.5B 共 24 层 → 290 张量与实测一致。

### 3.3 用 safetensors 加载（Context7 `/safetensors/safetensors` 核实的 API）

```python
from safetensors import safe_open
from safetensors.torch import load_file  # 便捷：整文件 -> dict[str, Tensor]

# 单文件（0.5B）：懒加载、零拷贝，按 key 取张量、可直接落 device
state = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for k in f.keys():
        state[k] = f.get_tensor(k)
# 等价便捷写法： state = load_file("model.safetensors", device="cpu")

# 分片（7B）：先读 index 的 weight_map，再按分片逐个 safe_open 合并
import json, pathlib
idx = json.loads(pathlib.Path("model.safetensors.index.json").read_text())
shards = sorted(set(idx["weight_map"].values()))
state = {}
for shard in shards:
    state.update(load_file(shard, device="cpu"))

# 灌进自实现模块：名字对齐后 load_state_dict（0.5B 记得处理 tied lm_head）
model.load_state_dict(state, strict=True)
```

- `safe_open(..., framework="pt", device=...)` 支持懒加载/零拷贝，`f.get_slice(key)` 可只取张量切片（大模型分卡时省内存）。
- 有共享张量时官方推荐 `safetensors.torch.load_model(model, path)` 而非手动 `load_state_dict`（tied embedding 属于这种情况，可作为备选路径）。

---

## 4. API 用法要点（对拍 / 加载，基于实测 transformers 4.56.2）

> ⚠️ 已按 ADR-0006 更正：**真实版本是 transformers 4.56.2（4.x）**，非 R2 所称的 5.15.0。原 §4 整段「transformers 5.x 破坏性变更」是对**不存在版本**的描述，已删除；下面只保留在 **4.56.2 上实测成立、且本仓代码依赖**的用法。

infrared 里 HF `transformers` 只作**权重来源 + 正确性对拍标尺**（ADR-0005），不是运行时执行路径（ADR-0003）。对拍要点（`tests/test_parity.py` 实测通过）：

- **加载精度用 `dtype=`**：`from_pretrained(..., dtype=torch.float32)`（4.56 已支持 `dtype=` 关键字；本仓 parity 用它，不用旧 `torch_dtype=`）。
- **对拍用 `attn_implementation="eager"`**：拿与朴素实现最接近的数值路径，配合 `dtype=float32`、greedy、固定 seed 比对首步 logits 与 greedy 生成。
- **绝不用 `.generate()`**：HF 仅作 oracle——自实现 forward 逐 token 生成后，与 HF 参考输出比对。

torch↔triton：`torch==2.12.0` 的 Linux CUDA wheel 自带并钉住其配对 triton（我们不自钉，见 §1）。自实现 kernel 走 Triton（T4）；朴素融合注意力可先用 `F.scaled_dot_product_attention`。

---

## 5. tokenizer 用法片段

Qwen2.5 用 BPE tokenizer（仓库含 `tokenizer.json` / `vocab.json` / `merges.txt` / `tokenizer_config.json`），ChatML 对话模板。经 Context7 `/huggingface/transformers` 核实。

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 对话：套 ChatML 模板并加生成引导（<|im_start|>assistant）
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "用一句话解释 GQA。"},
]
prompt = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
input_ids = tok(prompt, return_tensors="pt").input_ids  # -> 喂给自实现 forward

# 解码（跳过特殊符）
text = tok.decode(generated_ids[0], skip_special_tokens=True)
```

- 停止符：`eos_token_id = [151645, 151643]`（`<|im_end|>` 与 `<|endoftext|>`，来自 `generation_config.json`）。自实现 decode loop 命中任一即停。
- 0.5B 与 7B 的对话模板/特殊符一致，切模型只换 `from_pretrained` 的名字即可。

---

_关联 issue：[xiangzhang-coding/infrared#3](https://github.com/xiangzhang-coding/infrared/issues/3)（R2 · Research：较新稳定版依赖 + Qwen2.5 架构超参）。_
