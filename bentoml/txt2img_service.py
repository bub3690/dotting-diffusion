import bentoml
from bentoml.io import Image, JSON #, Multipart
from starlette.middleware.cors import CORSMiddleware
from runners.diffusion_runner import StableDiffusionRunnable




# TXT2IMG_MODEL_RUNNER=bentoml.diffusers.get(TXT2IMG_MODEL_TAG).to_runner()
# #TXT2IMG_MODEL_RUNNER = bentoml.diffusers.get_runnable(TXT2IMG_MODEL)
# print("here")
# #TXT2IMG_MODEL_RUNNER = bentoml.diffusers.load_model(TXT2IMG_MODEL).to_runner()
# art_diffusion= bentoml.Service("art_diffusion",runners=[TXT2IMG_MODEL_RUNNER])
# art_diffusion.add_asgi_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])



stable_diffusion_runner = bentoml.Runner(StableDiffusionRunnable, name='stable_diffusion_runner', max_batch_size=10)

stable_diffusion_fp16 = bentoml.Service("stable_diffusion_fp16", runners=[stable_diffusion_runner,])
stable_diffusion_fp16.add_asgi_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])


@stable_diffusion_fp16.api(input=JSON(),output=Image(mime_type="image/png"))
def txt2img(parsed_json):
    #print(parsed_json)
    image = stable_diffusion_runner.txt2img.run(parsed_json)
    return image





