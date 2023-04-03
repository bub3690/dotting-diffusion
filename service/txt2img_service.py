import bentoml
from bentoml.io import Image, JSON, Multipart
from starlette.middleware.cors import CORSMiddleware
import torch
from rembg import remove

from contextlib import ExitStack



# TXT2IMG_MODEL_RUNNER=bentoml.diffusers.get(TXT2IMG_MODEL_TAG).to_runner()
# #TXT2IMG_MODEL_RUNNER = bentoml.diffusers.get_runnable(TXT2IMG_MODEL)
# print("here")
# #TXT2IMG_MODEL_RUNNER = bentoml.diffusers.load_model(TXT2IMG_MODEL).to_runner()
# art_diffusion= bentoml.Service("art_diffusion",runners=[TXT2IMG_MODEL_RUNNER])
# art_diffusion.add_asgi_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])

class StableDiffusionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):

        TXT2IMG_MODEL_TAG = "txt2img_pixel_art_diffusion:latest"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.txt2img_pipe = bentoml.diffusers.load_model(TXT2IMG_MODEL_TAG,enable_xformers=False)#.to(self.device)
        #pipeline을 매번 만드느라 느린 단점이 있다.

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

stable_diffusion_runner = bentoml.Runner(StableDiffusionRunnable, name='stable_diffusion_runner', max_batch_size=10)

svc = bentoml.Service("stable_diffusion_fp16", runners=[stable_diffusion_runner])
svc.add_asgi_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])

@svc.api(input=JSON(),output=Image(mime_type="image/png"))
def txt2img(parsed_json):
    #print(parsed_json)
    image = stable_diffusion_runner.txt2img.run(parsed_json)
    return image





