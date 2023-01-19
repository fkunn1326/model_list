from huggingface_hub import list_models
import json
import aiohttp
import asyncio

models = list_models(filter="stable-diffusion", full=True)

"""
filterかけるやつ

- ckptとsafetensors以外
- diffusion_pytorch_model.xxx
"""

ml = []
l = []

async def main():
    async with aiohttp.ClientSession() as session:
        for model in models:
            for file in model.siblings:
                if file.rfilename.endswith("ckpt") or file.rfilename.endswith("safetensors"):
                    if not file.rfilename.split("/")[-1].startswith("diffusion_pytorch_model"):
                        if not "vae" in file.rfilename:
                            if not "safety_checker" in file.rfilename:
                                if not "text_encoder" in file.rfilename:
                                    if not "unet" in file.rfilename:
                                        idx = next((index for (index, d) in enumerate(ml) if d["model_repo"] == model.modelId), None)
                                        async with session.get(f"https://huggingface.co/{model.modelId}/raw/main/{file.rfilename}") as resp:
                                            try:
                                                text = await resp.text()
                                                hash = text.split("\n")[1].split(":")[1]
                                                if idx is not None:
                                                    ml[idx]["file_list"].append({
                                                        "file_name": file.rfilename,
                                                        "hash": hash
                                                    })
                                                else:
                                                    ml.append({
                                                        "model_repo": model.modelId,
                                                        "file_list": [{
                                                            "file_name": file.rfilename,
                                                            "hash": hash
                                                        }]
                                                    })
                                            except IndexError:
                                                print(f"https://huggingface.co/{model.modelId}/raw/main/{file.rfilename}")
asyncio.run(main())

with open("./public/data.json", 'w', encoding="utf8") as f:
    json.dump(ml, f, indent=4, ensure_ascii=False)