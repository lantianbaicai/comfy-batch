"""
第二轮优化：王家卫「欲」升级 + 杂志封面排版优化
"""
import requests, time, os, uuid, sys
from datetime import date

HOST = "127.0.0.1:8000"
OUTPUT_DIR = os.path.join(r"E:\opc-work", "comfy_output", date.today().isoformat())

PROMPTS_V2 = {
    # === 王家卫 v2 — 加「欲」：更亲密、更暗示、更张力 ===
    "wkw2_花样年华_欲": (
        "masterpiece, best quality, 8k, cinematic still photography, "
        "Wong Kar-wai In the Mood for Love aesthetic, shot by Christopher Doyle, "
        "extreme close-up, a woman's bare back and nape of neck, "
        "she wears a tight red cheongsam partially unzipped from behind, "
        "her skin glistening with a thin layer of sweat under dim amber light, "
        "a man's hand hovering inches from her bare shoulder, almost touching but not, "
        "narrow Hong Kong corridor, peeling wallpaper, single bare bulb swinging, "
        "rain streaming down window behind them, green neon reflecting on wet skin, "
        "shallow depth of field, 50mm f/0.95, bokeh rain drops, "
        "the tension of a touch that hasn't happened yet, sexual tension, desire, longing, "
        "film grain, step-printing, slightly underexposed, saturated reds and greens, "
        "tasteful artistic cinematography, implied sensuality, elegant eroticism, "
        "花樣年華, 暧昧, 渴望"
    ),
    "wkw2_重庆森林_欲": (
        "masterpiece, best quality, 8k, cinematic still photography, "
        "Wong Kar-wai Chungking Express aesthetic, shot by Christopher Doyle, "
        "a woman in a white tank top, slightly damp, fabric clinging to skin, "
        "lying on a messy bed, tangled white sheets barely covering her, "
        "one strap of her top slipped down revealing bare shoulder and collarbone, "
        "her eyes half-closed looking directly at camera with drowsy desire, "
        "blue aquarium light casting rippling shadows across her body, "
        "a half-empty glass of whiskey on nightstand, cigarette burning in ashtray, "
        "tiny beads of condensation on her skin, summer night humidity, "
        "wide angle lens, 24mm, distorted intimacy, getting too close, "
        "the feeling of 4am when you can't sleep and you want someone, "
        "film grain, motion blur on edges, green-blue color cast, neon outside window, "
        "tasteful artistic photography, implied nudity through sheets, sensual but elegant"
    ),
    "wkw2_堕落天使_欲": (
        "masterpiece, best quality, 8k, cinematic still photography, "
        "Wong Kar-wai Fallen Angels aesthetic, shot by Christopher Doyle, "
        "a woman in a black silk robe, sitting on edge of bathtub in dim bathroom, "
        "robe slipping off one shoulder revealing bare skin, steam fogging mirror, "
        "her reflection in the fogged mirror, partially obscured, more mysterious than real, "
        "wet hair clinging to her neck and cheek, water droplets on her collarbone, "
        "single red neon strip under the mirror casting hellish glow, "
        "she looks at her own reflection with a mix of contempt and desire, "
        "0.3 second shutter drag creating ghost image in mirror, motion blur, "
        "the mirror shows what the camera can't — the suggestion of more, "
        "fish eye lens distortion, claustrophobic framing, voyeuristic angle, "
        "film grain, green fluorescent and red neon color clash, "
        "tasteful artistic cinematography, implied nudity, desire meets self-loathing"
    ),

    # === 杂志封面 v2 — 加排版空间和杂志感 ===
    "mag2_vogue_asia": (
        "masterpiece, best quality, 8k, VOGUE magazine cover photograph, "
        "commercial editorial photography, magazine cover layout composed, "
        "a supermodel Asian woman in dramatic red haute couture gown, "
        "clean top 30 percent of image empty for VOGUE masthead text, "
        "clean left and right margins for cover lines, "
        "standing against pure white studio backdrop with single dramatic shadow, "
        "high fashion pose, one hand elegantly on hip, chin slightly raised, "
        "strong eye contact with camera, powerful editorial expression, "
        "beauty dish key light from above, sharp catchlights in eyes, "
        "the entire image designed as a magazine cover with clear hierarchy: "
        "model in center, empty space at top for title, sides for article teasers, "
        "Hasselblad H6D, 100mm f/2.2, commercial fashion photography, "
        "high contrast, vibrant red against pure white, editorial perfection, "
        "skin retouched but natural texture visible, luxury fashion aesthetic"
    ),
    "mag2_bazaar_cn": (
        "masterpiece, best quality, 8k, Harper's BAZAAR China magazine cover, "
        "editorial magazine photography, front cover composition, "
        "an elegant Asian celebrity in sculptural white avant-garde dress, "
        "top area reserved for BAZAAR masthead, right side open for cover lines, "
        "bottom area clean for barcode and date, "
        "dramatic studio lighting, split lighting half face in shadow half in light, "
        "minimalist grey backdrop, architectural composition, geometric shapes, "
        "high fashion art direction, the photo IS the magazine layout, "
        "phase One IQ4, medium format, sharp details, luxury editorial, "
        "Chinese fashion meets Western avant-garde, 时尚芭莎, "
        "graphic composition, bold negative space, magazine-ready"
    ),
    "mag2_cosmo_cover": (
        "masterpiece, best quality, 8k, Cosmopolitan magazine cover photograph, "
        "bright colorful editorial, magazine front cover layout, "
        "a smiling young Asian celebrity in vibrant yellow sundress, off-shoulder, "
        "top of image clean for COSMOPOLITAN masthead, "
        "happy, energetic, genuine laugh, wind in hair, golden hour backlight, "
        "tropical beach background with turquoise water, vacation vibe, "
        "cover lines space on both sides, bottom banner space for secondary stories, "
        "commercial magazine photography, accessible fashion, girl-next-door beauty, "
        "Sony A1, 35mm, bright and airy, summer cover, COSMO girl energy"
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
        "105": {"inputs": {"images": ["104", 0], "filename_prefix": "wkw_mag_v2"}, "class_type": "SaveImage"},
    }


def run(prompts_dict):
    for label, prompt in prompts_dict.items():
        wf = build_workflow(prompt)
        r = requests.post(f"http://{HOST}/prompt", json={"prompt": wf, "client_id": str(uuid.uuid4())}, timeout=10)
        pid = r.json()["prompt_id"]
        print(f"  [{label}] {pid[:8]}...")
        for _ in range(60):
            time.sleep(2)
            h = requests.get(f"http://{HOST}/history/{pid}", timeout=10).json()
            if pid in h:
                for _, node_out in h[pid]["outputs"].items():
                    for img in node_out.get("images", []):
                        fn = img["filename"]; sub = img.get("subfolder", "")
                        data = requests.get(f"http://{HOST}/view?filename={fn}&subfolder={sub}&type=output", timeout=30).content
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        out = os.path.join(OUTPUT_DIR, f"v2_{label}_{fn}")
                        with open(out, "wb") as f: f.write(data)
                        print(f"  ✅ [{label}] {len(data)//1024}KB")
                break
        else:
            print(f"  ❌ [{label}] 超时")
    print(f"\n  全部完成！{OUTPUT_DIR}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    print("=" * 60)
    print("  🎬 第二轮：欲升级 + 杂志排版")
    print("=" * 60)
    if mode in ("wkw", "both"):
        wkw = {k: v for k, v in PROMPTS_V2.items() if k.startswith("wkw")}
        print(f"\n  🌧️ 王家卫·欲 ({len(wkw)}张) — 更多暗示、触感、张力")
        run(wkw)
    if mode in ("mag", "both"):
        mag = {k: v for k, v in PROMPTS_V2.items() if k.startswith("mag")}
        print(f"\n  📰 杂志封面·排版 ({len(mag)}张) — 预留masthead+cover lines空间")
        run(mag)
