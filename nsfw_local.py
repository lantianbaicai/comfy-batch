"""
本地NSFW — 仅供本地观看，不上传任何平台
"""
import requests, time, os, uuid
from datetime import date

HOST = "127.0.0.1:8000"
OUTPUT_DIR = os.path.join(r"E:\opc-work", "comfy_output", date.today().isoformat(), "nsfw_local")

PROMPTS = {
    "nsfw_唐风_浴": (
        "masterpiece, best quality, 8k, artistic nude photography, "
        "a beautiful Asian woman bathing in a traditional Chinese wooden bathtub, "
        "rose petals floating on water surface, steam rising, "
        "her bare shoulders and upper chest visible above water, wet hair trailing, "
        "candlelit bathroom, red silk curtains, antique bronze mirror on wall, "
        "warm amber lighting, glowing skin, water droplets on collarbone, "
        "sensual and elegant, classical Chinese boudoir aesthetic, "
        "artistic nude, fine art photography, Helmut Newton meets Chinese erotica, "
        "no explicit content, tastefully composed, skin texture, natural breast shape visible above water"
    ),
    "nsfw_宋风_寝": (
        "masterpiece, best quality, 8k, artistic nude photography, "
        "an elegant Asian woman lying on antique Chinese bed, sheer white silk partially draped over body, "
        "moonlight through paper windows casting soft blue light across bare skin, "
        "her body visible through translucent fabric, side profile, curves highlighted by shadow, "
        "traditional Chinese bedroom, carved wooden bed frame, silk pillows, incense smoke, "
        "Classical Chinese erotic art meets modern fine art nude, 春宫画 inspired, "
        "tasteful nudity, breast and hip visible but artistically composed, "
        "Peter Lindbergh style lighting, moody and intimate, editorial nude"
    ),
    "nsfw_明风_镜": (
        "masterpiece, best quality, 8k, artistic nude photography, "
        "a beautiful woman in Ming Dynasty boudoir, standing nude before a full-length bronze mirror, "
        "her bare back and buttocks visible, reflection showing her front partially obscured by silk robe, "
        "she is in the act of dressing, red silk robe held loosely in one hand barely covering her, "
        "warm light from paper lantern, polished rosewood furniture, pearl ornaments on vanity, "
        "the mirror reveals what the camera hides — a glimpse of bare breasts in reflection, "
        "classical Chinese erotic painting aesthetic, 仕女图 meets modern nude, "
        "tasteful fine art photography, editorial quality, artistic nudity"
    ),
    "nsfw_唐风_榻": (
        "masterpiece, best quality, 8k, artistic nude photography, "
        "a sensual Asian woman nude on a traditional Chinese daybed, lying on her stomach, "
        "her bare back fully visible, curves of hips and buttocks subtly defined, "
        "she looks back over her bare shoulder at the camera with a knowing expression, "
        "red silk sheets tangled around her legs, peony flowers scattered on bed, "
        "dim amber lantern light, shadows playing across her bare skin, "
        "Tang Dynasty boudoir scene, classical eroticism, 海棠春睡, "
        "fine art nude photography, tasteful and artistic, skin texture visible, "
        "no explicit genitalia, artistic composition, editorial fashion nude edge"
    ),
    "nsfw_汉服_纱": (
        "masterpiece, best quality, 8k, artistic nude photography, "
        "a woman in a completely sheer white silk hanfu robe, body fully visible through translucent fabric, "
        "standing in morning light by an open window, backlit creating silhouette of her nude form, "
        "breasts and body contours clearly visible but softened by the fabric and light, "
        "her hair loose, natural pose, one hand gently holding robe closed at chest, "
        "traditional Chinese garden outside window, bamboo and lotus pond, morning mist, "
        "the sheer fabric is the only thing between her body and the viewer, "
        "classical Chinese aesthetic, 仕女画, artistic nude through fabric, "
        "fine art photography, soft natural light, ethereal and sensual"
    ),
}


def build_workflow(prompt, seed=None):
    if seed is None:
        seed = int(time.time() * 1000) % 999999999999
    return {
        "28": {"inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "30": {"inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}, "class_type": "CLIPLoader"},
        "29": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"},
        "11": {"inputs": {"model": ["28", 0], "shift": 3.0}, "class_type": "ModelSamplingAuraFlow"},
        "27": {"inputs": {"text": prompt, "clip": ["30", 0]}, "class_type": "CLIPTextEncode"},
        "33": {"inputs": {"conditioning": ["27", 0]}, "class_type": "ConditioningZeroOut"},
        "13": {"inputs": {"width": 768, "height": 1024, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
        "3": {"inputs": {"model": ["11", 0], "positive": ["27", 0], "negative": ["33", 0], "latent_image": ["13", 0], "seed": seed, "steps": 6, "cfg": 1.1, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}, "class_type": "KSampler"},
        "8": {"inputs": {"samples": ["3", 0], "vae": ["29", 0]}, "class_type": "VAEDecode"},
        "101": {"inputs": {"image": ["8", 0], "upscale_method": "lanczos", "width": 1024, "height": 1536, "crop": "center"}, "class_type": "ImageScale"},
        "102": {"inputs": {"pixels": ["101", 0], "vae": ["29", 0]}, "class_type": "VAEEncode"},
        "103": {"inputs": {"model": ["11", 0], "positive": ["27", 0], "negative": ["33", 0], "latent_image": ["102", 0], "seed": seed + 1, "steps": 4, "cfg": 1.1, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 0.35}, "class_type": "KSampler"},
        "104": {"inputs": {"samples": ["103", 0], "vae": ["29", 0]}, "class_type": "VAEDecode"},
        "105": {"inputs": {"images": ["104", 0], "filename_prefix": "nsfw_local"}, "class_type": "SaveImage"},
    }


for label, prompt in PROMPTS.items():
    wf = build_workflow(prompt)
    r = requests.post(f"http://{HOST}/prompt", json={"prompt": wf, "client_id": str(uuid.uuid4())}, timeout=10)
    pid = r.json()["prompt_id"]
    print(f"  [{label}] {pid[:8]}...")
    for _ in range(60):
        time.sleep(2)
        h = requests.get(f"http://{HOST}/history/{pid}", timeout=10).json()
        if pid in h:
            for _, out in h[pid]["outputs"].items():
                for img in out.get("images", []):
                    fn = img["filename"]; sub = img.get("subfolder", "")
                    data = requests.get(f"http://{HOST}/view?filename={fn}&subfolder={sub}&type=output", timeout=30).content
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    path = os.path.join(OUTPUT_DIR, f"{label}_{fn}")
                    with open(path, "wb") as f: f.write(data)
                    print(f"  ✅ [{label}] {path} ({len(data)//1024}KB)")
            break
    else:
        print(f"  ❌ [{label}] 超时")
print(f"\n  全部完成！{OUTPUT_DIR}")
