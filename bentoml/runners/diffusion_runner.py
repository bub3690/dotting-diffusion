import torch
import bentoml
from diffusers import DiffusionPipeline
from rembg import remove


class StableDiffusionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        # 이부분을 bentoml model로 교체해야함.
        TXT2IMG_MODEL_TAG = "PublicPrompts/All-In-One-Pixel-Model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.txt2img_pipe = DiffusionPipeline.from_pretrained(TXT2IMG_MODEL_TAG,torch_dtype=torch.float16).to(self.device)
        else:
            self.txt2img_pipe = DiffusionPipeline.from_pretrained(TXT2IMG_MODEL_TAG,torch_dtype=torch.float32).to(self.device)

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def txt2img(self, parsed_json):
        #print("hello")
        src_prompt = parsed_json.get("prompt") + ",pixelsprite,full body game asset,background chroma plain green"

        image = self.txt2img_pipe(src_prompt,
        guidance_scale=7.5,
        negative_prompt=",".join(['']),
        num_images_per_prompt=1,
        ).images
        #image = [ remove(sample) for sample in image]#png 출력
        image = remove(image[0])

        return image
    
    
    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def txt2img_multi(self, parsed_json)->list:
        print("multi")
        #여러 이미지를 출력하는 api
        src_prompt = parsed_json.get("prompt") + ",pixelsprite,full body game asset,background chroma plain green"

        num_imgs = parsed_json.get("num_imgs") # 서버 성능에 따라 최대 이미지 수는 달라질 것.
        if num_imgs is None:
            num_imgs = 2

        image = self.txt2img_pipe(src_prompt,
        guidance_scale=7.5,
        negative_prompt=",".join(['']),
        num_images_per_prompt=num_imgs,
        ).images
        
        images = [ remove(sample) for sample in image]#png 출력
        #image = remove(image[0])

        return images