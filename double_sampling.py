"""
Z-Image 双采样（Double Sampling）工作流
========================================
原理：第一遍生成结构(低分辨率) → 放大 → 第二遍补充细节(高分辨率+低denoise)

用法：
  python -X utf8 double_sampling.py "提示词" --nsfw

作者：云曦
"""
import requests, time, os, uuid, sys, json
from datetime import date

HOST = "127.0.0.1:8000"
OUTPUT_DIR = os.path.join(r"E:\opc-work", "comfy_output", date.today().isoformat())

# ============================================================
# 汉服写真提示词模板
# ============================================================
HANFU_PROMPTS = {
    "唐风仕女": (
        "masterpiece, best quality, ultra high resolution, 8k, professional photography, "
        "a beautiful Chinese woman in Tang Dynasty hanfu, red and gold silk dress with wide sleeves, "
        "traditional Tang dynasty makeup with flower pattern on forehead, "
        "standing in ancient palace courtyard with red lanterns, cherry blossoms falling, "
        "soft golden hour backlight, shallow depth of field, bokeh background, "
        "elegant pose, looking at viewer with gentle smile, "
        "professional portrait photography, Hasselblad, cinematic lighting, "
        "skin texture, natural skin, realistic photograph"
    ),
    "宋制素雅": (
        "masterpiece, best quality, ultra high resolution, 8k, professional photography, "
        "a graceful Chinese woman in Song Dynasty hanfu, light blue and white silk ruqun, "
        "simple elegant hair ornament with jade hairpin, light natural makeup, "
        "standing by a bamboo grove near a lotus pond, morning mist, soft diffused light, "
        "holding a round silk fan, looking downward thoughtfully, "
        "poetic atmosphere, Song dynasty aesthetics, minimalist elegance, "
        "professional portrait, Canon EOS R5, natural lighting, "
        "skin pores visible, realistic skin texture"
    ),
    "明制端庄": (
        "masterpiece, best quality, ultra high resolution, 8k, professional photography, "
        "a dignified Chinese woman in Ming Dynasty hanfu, deep blue aoqun with gold embroidery, "
        "elaborate phoenix hair crown, pearl earrings, mature elegant makeup with red lips, "
        "sitting in a classical Chinese study room, rosewood furniture, calligraphy scrolls on wall, "
        "warm candlelight, dramatic chiaroscuro lighting, "
        "royal court lady aura, looking directly at camera with quiet authority, "
        "medium format, Fujifilm GFX, soft shadows, "
        "hyper-detailed fabric texture, realistic photograph"
    ),
    "魏晋风骨": (
        "masterpiece, best quality, ultra high resolution, 8k, professional photography, "
        "an ethereal Chinese woman in Wei-Jin Dynasty hanfu, flowing white robes with wide sleeves, "
        "minimalist free-flowing hair, no heavy ornaments, natural no-makeup look, "
        "standing on a mountain cliff edge, misty mountains in background, wind blowing robes and hair, "
        "dramatic sky with breaking clouds, god rays, "
        "looking into distance with transcendent expression, "
        "poetic immortal aesthetic, xianxia atmosphere, "
        "professional landscape portrait, wide angle, epic composition"
    ),
    "唐风NSFW艺术": (
        "masterpiece, best quality, ultra high resolution, 8k, artistic nude photography, "
        "a beautiful woman in sheer Tang Dynasty inspired silk robes, translucent red fabric, "
        "lying on traditional Chinese daybed with silk cushions, peony flowers scattered, "
        "warm amber lighting from paper lanterns, smoke wisps in air, "
        "elegant artistic pose, partial silhouette, tasteful boudoir aesthetic, "
        "古典春宫画美学, 艺术人体, Helmut Newton style meets Chinese classical art, "
        "cinematic lighting, film grain, professional art photography, "
        "tasteful and artistic, no explicit content, fine art quality"
    ),
    "宋风NSFW艺术": (
        "masterpiece, best quality, ultra high resolution, 8k, artistic nude photography, "
        "an elegant woman in Song Dynasty inspired sheer white ruqun, translucent layers, "
        "bathing in hot spring with flower petals floating, steam rising, bamboo screen background, "
        "moonlight through paper windows, soft blue night lighting, "
        "artistic rear view, shoulder blades visible, wet hair trailing in water, "
        "宋词意境, 海棠春睡, classical Chinese erotic art aesthetics, "
        "fine art nude photography, Peter Lindbergh style, moody atmosphere, "
        "tasteful composition, editorial fashion nude"
    ),
    "明艳宫装": (
        "masterpiece, best quality, ultra high resolution, 8k, professional photography, "
        "a stunning Chinese woman in elaborate Ming Dynasty court dress, bright red and gold, "
        "phoenix crown with kingfisher feather ornaments, heavy court makeup, "
        "inside Forbidden City style palace hall, golden pillars, dragon carvings, "
        "dramatic side lighting from palace windows, dust particles in light beams, "
        "full body shot, standing proudly, long train of dress flowing behind, "
        "imperial concubine aesthetic, 甄嬛传 style, ultra detailed embroidery, "
        "Hasselblad H6D, professional lighting setup"
    ),
    "唐宫夜宴NSFW": (
        "masterpiece, best quality, ultra high resolution, 8k, artistic photography, "
        "a sensual woman in Tang Dynasty banquet scene, sheer silk robe half off shoulders, "
        "lying amidst silk cushions and wine cups, grapes and fruits scattered, "
        "dim candlelight, red silk curtains, incense smoke, "
        "artistic boudoir scene, 韩熙载夜宴图 aesthetic meets modern photography, "
        "partial nude, tastefully draped fabric, warm amber and red tones, "
        "Renaissance painting lighting, chiaroscuro, fine art nude, "
        "elegant and artistic, editorial quality"
    ),
}


def build_double_sampling_workflow(prompt: str, seed: int = None,
                                    first_w: int = 768, first_h: int = 1024,
                                    final_w: int = 1024, final_h: int = 1536,
                                    first_steps: int = 6, refine_steps: int = 4,
                                    cfg: float = 1.1, denoise: float = 0.35):
    """
    构建双采样工作流JSON：
    第一遍768x1024生成结构 → 放大到1024x1536 → 第二遍补充细节
    """
    if seed is None:
        seed = int(time.time() * 1000) % 999999999999

    # 节点编号规则：
    # 原始节点保留编号，新增节点用 100+
    # 共享节点：UNETLoader(28), CLIPLoader(30), VAELoader(29), ModelSamplingAuraFlow(11)
    # 第一遍：EmptyLatent(13, first_w x first_h), CLIPEncode(27), CondZeroOut(33), KSampler(3, denoise=1.0)
    # 放大：VAEDecode(8) → ImageScale(101) → VAEEncode(102)
    # 第二遍：KSampler(103, denoise=0.35, with scaled latent) → VAEDecode(104) → Save(105)

    workflow = {
        # === 共享模型加载 ===
        "28": {"inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "30": {"inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}, "class_type": "CLIPLoader"},
        "29": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"},
        "11": {"inputs": {"model": ["28", 0], "shift": 3.0}, "class_type": "ModelSamplingAuraFlow"},

        # === 提示词编码 ===
        "27": {"inputs": {"text": prompt, "clip": ["30", 0]}, "class_type": "CLIPTextEncode"},
        "33": {"inputs": {"conditioning": ["27", 0]}, "class_type": "ConditioningZeroOut"},

        # === 第一遍：结构生成（低分辨率） ===
        "13": {"inputs": {"width": first_w, "height": first_h, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
        "3": {"inputs": {
            "model": ["11", 0], "positive": ["27", 0], "negative": ["33", 0],
            "latent_image": ["13", 0], "seed": seed,
            "steps": first_steps, "cfg": cfg,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0
        }, "class_type": "KSampler"},

        # === 第一遍解码 ===
        "8": {"inputs": {"samples": ["3", 0], "vae": ["29", 0]}, "class_type": "VAEDecode"},

        # === 图像放大 ===
        "101": {"inputs": {
            "image": ["8", 0],
            "upscale_method": "lanczos",
            "width": final_w,
            "height": final_h,
            "crop": "center"
        }, "class_type": "ImageScale"},

        # === 放大后重新编码回潜空间 ===
        "102": {"inputs": {"pixels": ["101", 0], "vae": ["29", 0]}, "class_type": "VAEEncode"},

        # === 第二遍：细节补充（高分辨率，低denoise） ===
        "103": {"inputs": {
            "model": ["11", 0], "positive": ["27", 0], "negative": ["33", 0],
            "latent_image": ["102", 0], "seed": seed + 1,
            "steps": refine_steps, "cfg": cfg,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": denoise
        }, "class_type": "KSampler"},

        # === 最终解码+保存 ===
        "104": {"inputs": {"samples": ["103", 0], "vae": ["29", 0]}, "class_type": "VAEDecode"},
        "105": {"inputs": {"images": ["104", 0], "filename_prefix": "double_sample"}, "class_type": "SaveImage"},
    }

    return workflow


def generate(prompt: str, label: str = "img", nsfw: bool = False, **kwargs):
    """提交双采样工作流到ComfyUI并等待完成"""
    workflow = build_double_sampling_workflow(prompt, **kwargs)

    r = requests.post(
        f"http://{HOST}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=10
    )
    prompt_id = r.json()["prompt_id"]
    print(f"  [{label}] 已提交 prompt_id={prompt_id[:8]}... 双采样中(预计30-60秒)")

    for _ in range(60):
        time.sleep(2)
        h = requests.get(f"http://{HOST}/history/{prompt_id}", timeout=10).json()
        if prompt_id in h:
            images = []
            for _, node_out in h[prompt_id]["outputs"].items():
                for img in node_out.get("images", []):
                    images.append(img)

            results = []
            for img in images:
                fname = img["filename"]
                sub = img.get("subfolder", "")
                img_url = f"http://{HOST}/view?filename={fname}&subfolder={sub}&type=output"
                img_data = requests.get(img_url, timeout=30).content

                os.makedirs(OUTPUT_DIR, exist_ok=True)
                out_path = os.path.join(OUTPUT_DIR, f"ds_{label}_{fname}")
                with open(out_path, "wb") as f:
                    f.write(img_data)
                results.append(out_path)
                print(f"  ✅ [{label}] {out_path} ({len(img_data)//1024}KB)")
            return results

    raise TimeoutError(f"[{label}] 超时")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "hanfu"

    print("=" * 60)
    print("  🎨 Z-Image 双采样工作流")
    print("=" * 60)
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  流程: 768x1024(6步结构) → 放大 → 1024x1536(4步细节, denoise=0.35)")
    print()

    prompts_to_run = {}

    if mode == "hanfu" or mode == "all":
        # 汉服写真（安全向）
        prompts_to_run.update({
            "唐风仕女": HANFU_PROMPTS["唐风仕女"],
            "宋制素雅": HANFU_PROMPTS["宋制素雅"],
            "明制端庄": HANFU_PROMPTS["明制端庄"],
            "魏晋风骨": HANFU_PROMPTS["魏晋风骨"],
            "明艳宫装": HANFU_PROMPTS["明艳宫装"],
        })

    if mode == "nsfw" or mode == "all":
        prompts_to_run.update({
            "唐风NSFW": HANFU_PROMPTS["唐风NSFW艺术"],
            "宋风NSFW": HANFU_PROMPTS["宋风NSFW艺术"],
            "唐宫夜宴NSFW": HANFU_PROMPTS["唐宫夜宴NSFW"],
        })

    if mode == "single":
        prompt = sys.argv[2] if len(sys.argv) > 2 else HANFU_PROMPTS["唐风仕女"]
        generate(prompt, "test")
        return

    for label, prompt in prompts_to_run.items():
        try:
            generate(prompt, label)
        except Exception as e:
            print(f"  ❌ [{label}] 失败: {e}")

    print(f"\n  全部完成！图片在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
