"""
ComfyBatch - One-line ComfyUI image generation
================================================
Usage:
    from comfy_batch import ZImage

    z = ZImage()
    z.generate("a cute cat, anime style")
    z.generate(["prompt1", "prompt2"], steps=6, cfg=1.1)
"""

import requests
import time
import os
import uuid
from typing import List, Optional, Union
from pathlib import Path


class ZImage:
    """Z-Image Turbo batch generator via ComfyUI API."""

    def __init__(
        self,
        host: str = "127.0.0.1:8000",
        unet: str = "z_image_turbo_bf16.safetensors",
        clip: str = "qwen_3_4b.safetensors",
        vae: str = "ae.safetensors",
        width: int = 1024,
        height: int = 1024,
        steps: int = 6,
        cfg: float = 1.1,
        sampler: str = "res_multistep",
        scheduler: str = "simple",
        shift: float = 3.0,
    ):
        self.host = host.rstrip("/")
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler
        self.shift = shift

    def _submit(self, prompt: str, seed: Optional[int] = None) -> str:
        if seed is None:
            seed = int(time.time() * 1000) % 999999999999

        workflow = {
            "28": {"inputs": {"unet_name": self.unet, "weight_dtype": "default"}, "class_type": "UNETLoader"},
            "30": {"inputs": {"clip_name": self.clip, "type": "lumina2", "device": "default"}, "class_type": "CLIPLoader"},
            "29": {"inputs": {"vae_name": self.vae}, "class_type": "VAELoader"},
            "27": {"inputs": {"text": prompt, "clip": ["30", 0]}, "class_type": "CLIPTextEncode"},
            "33": {"inputs": {"conditioning": ["27", 0]}, "class_type": "ConditioningZeroOut"},
            "13": {"inputs": {"width": self.width, "height": self.height, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
            "11": {"inputs": {"model": ["28", 0], "shift": self.shift}, "class_type": "ModelSamplingAuraFlow"},
            "3": {"inputs": {
                "model": ["11", 0], "positive": ["27", 0], "negative": ["33", 0],
                "latent_image": ["13", 0], "seed": seed,
                "steps": self.steps, "cfg": self.cfg,
                "sampler_name": self.sampler, "scheduler": self.scheduler, "denoise": 1.0
            }, "class_type": "KSampler"},
            "8": {"inputs": {"samples": ["3", 0], "vae": ["29", 0]}, "class_type": "VAEDecode"},
            "9": {"inputs": {"images": ["8", 0], "filename_prefix": "comfy_batch"}, "class_type": "SaveImage"}
        }

        r = requests.post(
            f"http://{self.host}/prompt",
            json={"prompt": workflow, "client_id": str(uuid.uuid4())},
            timeout=10
        )
        return r.json()["prompt_id"]

    def _wait(self, prompt_id: str, timeout: int = 120) -> List[dict]:
        for _ in range(timeout // 2):
            time.sleep(2)
            h = requests.get(f"http://{self.host}/history/{prompt_id}", timeout=10).json()
            if prompt_id in h:
                images = []
                for _, node_out in h[prompt_id]["outputs"].items():
                    for img in node_out.get("images", []):
                        images.append(img)
                return images
        raise TimeoutError(f"Generation timed out after {timeout}s")

    def generate(
        self,
        prompts: Union[str, List[str]],
        output_dir: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Generate images from one or more prompts.

        Args:
            prompts: A single prompt string or list of prompt strings.
            output_dir: Directory to save images. Default: E:\opc-work\comfy_output\YYYY-MM-DD\
            **kwargs: Override any init parameter (steps, cfg, width, height, etc.)

        Returns:
            List of saved image file paths.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Default output dir
        if output_dir is None:
            from datetime import date
            output_dir = os.path.join(r'E:\opc-work', 'comfy_output', date.today().isoformat())

        # Apply kwargs overrides
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        results = []
        for i, prompt in enumerate(prompts):
            pid = self._submit(prompt)
            images = self._wait(pid)

            for img in images:
                fname = img["filename"]
                sub = img.get("subfolder", "")
                img_url = f"http://{self.host}/view?filename={fname}&subfolder={sub}&type=output"
                img_data = requests.get(img_url, timeout=30).content

                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, fname)

                with open(out_path, "wb") as f:
                    f.write(img_data)
                results.append(out_path)
                print(f"[{i+1}/{len(prompts)}] {out_path} ({len(img_data)//1024}KB)")

        return results


# Quick CLI entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m comfy_batch 'your prompt here'")
        print("       python -m comfy_batch 'prompt1' 'prompt2' 'prompt3'")
        sys.exit(1)

    z = ZImage()
    prompts = sys.argv[1:]
    z.generate(prompts)
