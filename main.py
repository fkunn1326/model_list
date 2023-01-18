from huggingface_hub import list_models
import json

models = list_models(filter=("stable-diffusion"))

# with open(r"C:\Users\fukuda\Desktop\test.txt", "w", encoding='UTF-8') as f:
#     print(models, file=f)

"""
filterかけるやつ

- ckptとsafetensors以外
- diffusion_pytorch_model.xxx
"""

ml = []
l = []

for model in models:
    for file in model.siblings:
        if file.rfilename.endswith("ckpt") or file.rfilename.endswith("safetensors"):
            if not file.rfilename.split("/")[-1].startswith("diffusion_pytorch_model"):
                if not "vae" in file.rfilename:
                    if not "safety_checker" in file.rfilename:
                        if not "text_encoder" in file.rfilename:
                            if not "unet" in file.rfilename:
                                idx = next((index for (index, d) in enumerate(ml) if d["model_repo"] == model.modelId), None)
                                if idx is not None:
                                    ml[idx]["file_list"].append(file.rfilename)
                                else:
                                    ml.append({
                                        "model_repo": model.modelId,
                                        "file_list": [file.rfilename]
                                    })

with open(r"data.json", 'w', encoding="utf8") as f:
    json.dump(ml, f, indent=4, ensure_ascii=False)
