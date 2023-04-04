import torch
import bentoml
from diffusers import DiffusionPipeline
from rembg import remove


class StableDiffusionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):

        TXT2IMG_MODEL_TAG = "PublicPrompts/All-In-One-Pixel-Model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.txt2img_pipe = DiffusionPipeline.from_pretrained(TXT2IMG_MODEL_TAG,torch_dtype=torch.float16).to(self.device)
        else:
            self.txt2img_pipe = DiffusionPipeline.from_pretrained(TXT2IMG_MODEL_TAG,torch_dtype=torch.float32).to(self.device)

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def txt2img(self, parsed_json):
        print("hello")
        src_prompt = parsed_json.get("prompt") + ",pixelsprite,full body game asset,background chroma plain green"

        image = self.txt2img_pipe(src_prompt,
        guidance_scale=7.5,
        negative_prompt=",".join(['']),
        ).images[0]
        image = remove(image)#png 출력
        
        return image