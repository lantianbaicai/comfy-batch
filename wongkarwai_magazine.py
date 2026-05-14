"""
王家卫 + 杂志封面 双采样
python -X utf8 wongkarwai_magazine.py [wkw|mag|both]
"""
import requests, time, os, uuid, sys
from datetime import date

HOST = "127.0.0.1:8000"
OUTPUT_DIR = os.path.join(r"E:\opc-work", "comfy_output", date.today().isoformat())

PROMPTS = {
    # === 王家卫风格 — 似漏非漏、暧昧质感 ===
    "wkw_花样年华": (
        "masterpiece, best quality, 8k, cinematic photography, "
        "Wong Kar-wai film still, Chungking Express aesthetic, shot by Christopher Doyle, "
        "a beautiful Asian woman in a tight red cheongsam, sheer back with delicate lace trim, "
        "sitting by a rain-streaked window in a narrow Hong Kong corridor, "
        "neon green and red light bleeding through wet glass, cigarette smoke curling in air, "
        "step-printing motion blur, saturated colors, film grain, "
        "her bare shoulders visible through translucent fabric, elegant silhouette, "
        "looking away from camera with melancholic expression, "
        "intimate atmosphere, 花樣年華, moody lighting, shadows on wall, "
        "tasteful artistic photography, editorial fashion, "
        "skin texture, film stock, Kodak Portra 400"
    ),
    "wkw_重庆森林": (
        "masterpiece, best quality, 8k, cinematic photography, "
        "Wong Kar-wai film aesthetic, Fallen Angels style, shot by Christopher Doyle, "
        "a woman in a black silk slip dress, one strap fallen off shoulder, "
        "lying on a bed in a cramped Hong Kong apartment, fish tank casting blue light, "
        "wet hair, mascara slightly smeared, vulnerability captured in a fleeting moment, "
        "wide angle lens distortion, 0.5 second shutter drag, motion trails, "
        "neon signs outside window reflecting on her skin, green and purple color cast, "
        "her bare back visible, sheets tangled around legs, implied nudity, "
        "tasteful boudoir photography, 重庆森林, intimate but not explicit, "
        "film grain, 35mm, underexposed aesthetic"
    ),
    "wkw_2046": (
        "masterpiece, best quality, 8k, cinematic photography, "
        "Wong Kar-wai 2046 aesthetic, futuristic nostalgia, rich saturated colors, "
        "a woman in a sheer white silk robe, backlit by a single lamp in dark room, "
        "her silhouette visible through translucent fabric, lace details, "
        "sitting at a vintage vanity mirror, multiple reflections, "
        "red velvet curtains, gold trim, art deco furniture, "
        "smoke haze, shallow depth of field, 85mm f/1.2, "
        "her bare shoulder and collarbone elegantly exposed, fingers touching her neck, "
        "looking at her own reflection with desire and regret, "
        "tasteful artistic nude aesthetics, implied sensuality, "
        "Kodak Ektar 100, cinematic color grading, film still"
    ),

    # === 杂志封面风格 ===
    "mag_vogue_cn": (
        "masterpiece, best quality, 8k, professional magazine cover photography, "
        "VOGUE China cover style, fashion editorial, "
        "a stunning Asian woman in haute couture red silk gown, dramatic train, "
        "standing in empty Forbidden City courtyard at golden hour, "
        "strong editorial lighting, beauty dish key light, rim light on hair, "
        "looking directly at camera with powerful gaze, one hand on hip, "
        "top third of image empty for magazine masthead, "
        "clean commercial photography, Hasselblad H6D, 80mm f/2.8, "
        "fashion cover layout ready, negative space at top and sides, "
        "professional model pose, VOGUE aesthetic, high fashion, "
        "skin texture, natural beauty, Chinese fashion icon"
    ),
    "mag_elle_hk": (
        "masterpiece, best quality, 8k, professional magazine cover photography, "
        "ELLE Hong Kong cover style, urban fashion, "
        "a chic Asian woman in tailored black blazer over lace bralette, high-waist trousers, "
        "leaning against neon-lit wall in Mongkok night market, "
        "street fashion editorial, mixed ambient and flash lighting, "
        "top area reserved for magazine title, cover lines space on sides, "
        "candid editorial moment, wind blowing hair, "
        "Cosmopolitan magazine aesthetic, fashion forward, confident attitude, "
        "Canon EOS R5, 50mm f/1.2, editorial retouching"
    ),
    "mag_harper_bazaar": (
        "masterpiece, best quality, 8k, professional magazine cover photography, "
        "Harper's Bazaar China cover style, artistic fashion, "
        "an elegant Asian woman in flowing white silk dress, dramatic cape sleeves, "
        "standing in minimalist studio with dramatic single light source, "
        "high contrast black and white with selective red color accent on lips, "
        "top third clean for masthead, bold typography space on right side, "
        "avant-garde pose, arms creating geometric shapes, "
        "艺术时尚, 时尚芭莎, Peter Lindbergh meets Chinese aesthetics, "
        "medium format, Phase One, dramatic shadows, architectural composition"
    ),
    "mag_cosmo": (
        "masterpiece, best quality, 8k, professional magazine cover photography, "
        "Cosmopolitan magazine cover style, fresh and vibrant, "
        "a young Asian woman in colorful summer dress, off-shoulder, "
        "walking through Shanghai French Concession street, plane trees, morning light, "
        "bright natural lighting, golden hour glow, lens flare, "
        "top area clean for Cosmo masthead, colorful cover lines space, "
        "genuine laughter, hair flowing, candid movement, "
        "fresh editorial, youth fashion, COSMO girl aesthetic, "
        "Sony A1, 35mm f/1.4, bright and airy color grading"
    ),
}


def build_workflow(prompt, seed=None, first_w=768, first_h=1024, final_w=1024, final_h=1536):
    if seed is None:
        seed = int(time.time() * 1000) % 999999999999

    return {
        "28": {"inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "30": {"inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}, "class_type": "CLIPLoader"},
        "29": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"},
        "11": {"inputs": {"model": ["28", 0], "shift": 3.0}, "class_type": "ModelSamplingAuraFlow"},
        "27": {"inputs": {"text": prompt, "clip": ["30", 0]}, "class_type": "CLIPTextEncode"},
        "33": {"inputs": {"conditioning": ["27", 0]}, "class_type": "ConditioningZeroOut"},
        "13": {"inputs": {"width": first_w, "height": first_h, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
        "3": {"inputs": {"model": ["11", 0], "positive": ["27", 0], "negative": ["33", 0], "latent_image": ["13", 0], "seed": seed, "steps": 6, "cfg": 1.1, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}, "class_type": "KSampler"},
        "8": {"inputs": {"samples": ["3", 0], "vae": ["29", 0]}, "class_type": "VAEDecode"},
        "101": {"inputs": {"image": ["8", 0], "upscale_method": "lanczos", "width": final_w, "height": final_h, "crop": "center"}, "class_type": "ImageScale"},
        "102": {"inputs": {"pixels": ["101", 0], "vae": ["29", 0]}, "class_type": "VAEEncode"},
        "103": {"inputs": {"model": ["11", 0], "positive": ["27", 0], "negative": ["33", 0], "latent_image": ["102", 0], "seed": seed + 1, "steps": 4, "cfg": 1.1, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 0.35}, "class_type": "KSampler"},
        "104": {"inputs": {"samples": ["103", 0], "vae": ["29", 0]}, "class_type": "VAEDecode"},
        "105": {"inputs": {"images": ["104", 0], "filename_prefix": "wkw_mag"}, "class_type": "SaveImage"},
    }


def run(prompts_dict):
    for label, prompt in prompts_dict.items():
        wf = build_workflow(prompt)
        r = requests.post(f"http://{HOST}/prompt", json={"prompt": wf, "client_id": str(uuid.uuid4())}, timeout=10)
        pid = r.json()["prompt_id"]
        print(f"  [{label}] 提交 {pid[:8]}...")
        
        for _ in range(60):
            time.sleep(2)
            h = requests.get(f"http://{HOST}/history/{pid}", timeout=10).json()
            if pid in h:
                for _, node_out in h[pid]["outputs"].items():
                    for img in node_out.get("images", []):
                        fname = img["filename"]
                        sub = img.get("subfolder", "")
                        img_url = f"http://{HOST}/view?filename={fname}&subfolder={sub}&type=output"
                        data = requests.get(img_url, timeout=30).content
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        out = os.path.join(OUTPUT_DIR, f"ds_{label}_{fname}")
                        with open(out, "wb") as f:
                            f.write(data)
                        print(f"  ✅ [{label}] {out} ({len(data)//1024}KB)")
                break
        else:
            print(f"  ❌ [{label}] 超时")
    print(f"\n  全部完成！{OUTPUT_DIR}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    
    print("=" * 60)
    print("  🎬 王家卫 + 杂志封面 · 双采样")
    print("=" * 60)
    
    if mode in ("wkw", "both"):
        wkw = {k: v for k, v in PROMPTS.items() if k.startswith("wkw")}
        print(f"\n  🌧️ 王家卫风格 ({len(wkw)}张)")
        run(wkw)
    
    if mode in ("mag", "both"):
        mag = {k: v for k, v in PROMPTS.items() if k.startswith("mag")}
        print(f"\n  📰 杂志封面 ({len(mag)}张)")
        run(mag)
