# comfy-batch

一行代码让 ComfyUI 批量出图的 Python 工具。本来是自己用的，后来发现每次都手写 50 行 JSON 太傻，就打包了一下。

## 能做什么

```python
from comfy_batch import ZImage

z = ZImage()

# 一张
z.generate("a purple cat girl, chibi style")

# 一批
z.generate([
    "cyberpunk character, neon, 3D octane render",
    "watercolor cloud fairy, soft ethereal colors", 
    "tilt-shift diorama map of Bangkok",
])

# 双采样模式（结构→细节，质量翻倍但多花20秒）
z.generate("hanfu portrait, tang dynasty", double_sample=True)
```

## 安装

```bash
pip install comfy-batch
```

前提：ComfyUI 已启动（默认端口 8000），装好 Z-Image Turbo + Qwen 3 4B CLIP + ae VAE。

## 和手动写 API 的区别

之前用 ComfyUI 出图要走三步：写 50 行 JSON 拼工作流 → Python 调接口 → 手动解析结果下载。批量出图要套循环。

现在一行搞定。主要是省掉了拼 JSON 那步——不是不能手写，是真的没必要。

## 已知问题

- 一次别跑超过 6 张，显存会满（12GB 卡的上限）
- 不支持并发，ComfyUI 单实例串着跑
- 目前只适配 Z-Image Turbo，Flux 系的懒得加了

## 命令行也能用

```bash
python -m comfy_batch "a cat on a cloud"
python -m comfy_batch "prompt1" "prompt2" "prompt3"
```

## 配套模板

14套验证过的提示词模板见 [comfyui-hanfu-prompts](https://github.com/lantianbaicai/comfyui-hanfu-prompts)

---

MIT · star 随意，有问题提 issue。
