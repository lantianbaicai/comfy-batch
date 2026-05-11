# ComfyBatch 🎨

> **一行代码，批量出图。** 把 ComfyUI 从拖节点变成写代码。

[![Stars](https://img.shields.io/github/stars/lantianbaicai/comfy-batch?style=flat)](https://github.com/lantianbaicai/comfy-batch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## 这是什么

ComfyUI 有几百万用户，99% 在网页里手动拖节点。剩下 1% 写 Python 调 API 的，每次都要拼 50 行 JSON。

ComfyBatch 把这一切变成一行代码。

```python
from comfy_batch import ZImage

z = ZImage()

# 一张图
z.generate("a cute purple cat, anime style, 8K")

# 批量出图
z.generate([
    "beautiful diorama city map of Bangkok, tilt-shift, realistic",
    "cyberpunk female character, neon lights, 3D render, octane",
    "watercolor painting of cloud fairy, ethereal, soft colors",
])

# 自定义参数
z.generate("high quality portrait", steps=8, cfg=1.2, width=768, height=1024)
```

---

## 安装

```bash
pip install comfy-batch
```

**前提条件**：ComfyUI 已启动（默认 `127.0.0.1:8000`），并加载以下模型：

| 模型 | 路径 |
|------|------|
| Z-Image Turbo | `diffusion_models/z_image_turbo_bf16.safetensors` |
| Qwen 3 4B | `text_encoders/qwen_3_4b.safetensors` |
| VAE | `vae/ae.safetensors` |

连接远程 ComfyUI：
```python
z = ZImage(host="192.168.1.100:8000")
```

---

## 为什么用这个

| | ComfyBatch | 手动写 API | ComfyUI 网页 |
|---|---|---|---|
| 代码量 | **1 行** | ~50 行 | 0（拖节点） |
| 批量出图 | ✅ 一行搞定 | 😰 循环调接口 | ❌ 一张张点 |
| 可编程 | ✅ Python 原生 | ✅ 但繁琐 | ❌ |
| pip 安装 | ✅ | — | — |
| 适合场景 | 批量生产、自动化 | 定制化 | 偶尔用用 |

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `host` | `127.0.0.1:8000` | ComfyUI 地址 |
| `width` / `height` | 1024 | 出图尺寸 |
| `steps` | 6 | 采样步数（4-8 推荐） |
| `cfg` | 1.1 | CFG 引导强度 |
| `sampler` | `res_multistep` | 采样器 |
| `shift` | 3.0 | AuraFlow shift 参数 |

---

## 注意事项

- **显存管理**：连续出图建议每次不超过 6 张，避免显存溢出
- **并发**：不支持同时跑多张——ComfyUI 单实例串行处理
- **模型**：目前适配 Z-Image Turbo + Qwen CLIP 组合，Flux 等其他模型待后续支持

---

## 命令行模式

```bash
python -m comfy_batch "a beautiful sunset over mountains"
python -m comfy_batch "prompt1" "prompt2" "prompt3"
```

---

## License

MIT · 自由使用、修改、分发
