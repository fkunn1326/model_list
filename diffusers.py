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

def validate(str: str) -> str:
    allow_list = [
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    for item in allow_list:
        if item in str:
            return True
    return False

async def main():
    async with aiohttp.ClientSession() as session:
        for model in models:
            for file in model.siblings:
                if validate(file.rfilename):
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

with open("./public/diffusers.json", 'w', encoding="utf8") as f:
    json.dump(ml, f, indent=4, ensure_ascii=False)