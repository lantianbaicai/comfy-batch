# ComfyBatch 🎨

> One line. Any prompt. Batch generation via ComfyUI Z-Image Turbo.

```python
from comfy_batch import ZImage

z = ZImage()  # defaults to localhost:8000

# Single image
z.generate("a cute purple cat, anime style, 8K")

# Batch generation
z.generate([
    "a beautiful diorama city map of Bangkok, tilt-shift",
    "a cyberpunk character, neon lights, 3D render",
    "a watercolor painting of a cloud fairy, ethereal",
])

# Advanced
z.generate("high quality portrait", steps=8, cfg=1.2, width=512, height=768)
```

## Install

```bash
pip install comfy-batch
```

**Prerequisites:** ComfyUI running with Z-Image Turbo model loaded.

Required models (place in `ComfyUI/models/`):
- `diffusion_models/z_image_turbo_bf16.safetensors`
- `text_encoders/qwen_3_4b.safetensors`
- `vae/ae.safetensors`

## Why ComfyBatch?

ComfyUI has millions of users. 99% manually drag nodes in the web UI. Less than 1% write code to call its API.

**ComfyBatch lets you generate images from Python in one line.** No JSON workflow files. No node IDs. No API boilerplate.

## Features

- ✅ One-line generation
- ✅ Batch multiple prompts
- ✅ All parameters tunable (steps, cfg, resolution, sampler)
- ✅ Auto-download output images
- ✅ CLI mode: `python -m comfy_batch "your prompt"`
- ✅ Works with any ComfyUI instance (local or remote)

## Comparison

| | ComfyBatch | Manual API | ComfyUI Web UI |
|---|---|---|---|
| Lines of code | 1 | ~50 | 0 (drag nodes) |
| Batch generation | ✅ | 😰 | ❌ (one at a time) |
| Programmable | ✅ | ✅ | ❌ |
| pip installable | ✅ | — | — |

## License

MIT
