"""
GCP Storage version

"""
import bentoml
from bentoml.io import Image, JSON, Multipart
from starlette.middleware.cors import CORSMiddleware
from runners.diffusion_runner import StableDiffusionRunnable
from runners.image_remover_runner import ImageRemoverRunnable

from io import BytesIO





# TXT2IMG_MODEL_RUNNER=bentoml.diffusers.get(TXT2IMG_MODEL_TAG).to_runner()
# #TXT2IMG_MODEL_RUNNER = bentoml.diffusers.get_runnable(TXT2IMG_MODEL)
# print("here")
# #TXT2IMG_MODEL_RUNNER = bentoml.diffusers.load_model(TXT2IMG_MODEL).to_runner()
# art_diffusion= bentoml.Service("art_diffusion",runners=[TXT2IMG_MODEL_RUNNER])
# art_diffusion.add_asgi_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])



stable_diffusion_runner = bentoml.Runner(StableDiffusionRunnable, name='stable_diffusion_runner', max_batch_size=10)

background_remover_runner = bentoml.Runner(ImageRemoverRunnable, name='background_remover_runner', max_batch_size=10)

stable_diffusion_fp16 = bentoml.Service("stable_diffusion_fp16", runners=[stable_diffusion_runner,background_remover_runner])
stable_diffusion_fp16.add_asgi_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])



# 임시방편. Need to change Variable MultipleOutput
txt2img_output_format = Multipart(Image1=Image(mime_type="image/png"),
                                  Image2=Image(mime_type="image/png"),) #기본 2개 이미지.
 

@stable_diffusion_fp16.api(input=JSON(),output=txt2img_output_format)
def txt2img(parsed_json):
    #print(parsed_json)
    image = stable_diffusion_runner.txt2img.run(parsed_json) #List 형태
    files = dict()
    for i, sample in enumerate(image):
        file_name = f'Image{i+1}'
        files[file_name] = sample
    
    # for i in range(len(image),6):
    #     file_name = f'Image{i+1}'
    #     files[file_name] = None

    #print(files.keys())
    return files



@stable_diffusion_fp16.api(input=Image(),output=Image(mime_type="image/png"))
def background_remover(input_image):
    #print(parsed_json)
    image = background_remover_runner.background_remover.run(input_image)
    return image


