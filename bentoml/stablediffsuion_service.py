import bentoml
from bentoml.io import Image, JSON, Multipart
from starlette.middleware.cors import CORSMiddleware
from runners.diffusion_runner import StableDiffusionRunnable
from runners.image_remover_runner import ImageRemoverRunnable
import json

from io import BytesIO
from google.cloud import storage


import random
import string
import time

# 현재 시간을 기반으로 파일 이름을 생성합니다.
# 파일 이름은 현재 시간(초단위)을 기준으로 생성됩니다.
def generate_filename():
    # 현재 시간(초)을 가져옵니다.
    current_time = int(time.time())

    # 무작위로 구성된 알파벳과 숫자로 구성된 문자열을 생성합니다.
    # 문자열의 길이는 8자리로 제한합니다.
    random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    # 현재 시간(초)과 무작위 문자열을 조합하여 파일 이름을 생성합니다.
    filename = f"{current_time}_{random_chars}"
    
    return filename





def upload_blob(bucket_name, image_file):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    image_file_name = generate_filename()#난수로 생성시키기

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(image_file_name)
    
    with BytesIO() as output:
        image_file.save(output, format='PNG')
        contents = output.getvalue()
        
    blob.upload_from_string(contents, content_type='image/png')
    # Print the URL of the file
    print(
        f"File uploaded to {blob.public_url}."
    )
    return blob.public_url




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


@stable_diffusion_fp16.api(input=JSON(),output=Image(mime_type="image/png"))
def txt2img(parsed_json):
    #print(parsed_json)
    image = stable_diffusion_runner.txt2img.run(parsed_json) #List 형태
    
    # for i in range(len(image),6):
    #     file_name = f'Image{i+1}'
    #     files[file_name] = None

    #print(files.keys())
    return image


@stable_diffusion_fp16.api(input=JSON(),output=JSON())
def txt2img_multi(parsed_json):
    #output : image path
    files = dict()
    
    print(parsed_json)
    image_list = stable_diffusion_runner.txt2img_multi.run(parsed_json) #List 형태
    
    # upload image to gcp storage
    image_list = [ upload_blob("genai-bucket",image) for image in image_list ]#리스트 하나씩 인풋
    
    
    for i in range(len(image_list)):
        file_name = f'Image{i+1}'
        files[file_name] = image_list[i]

    print(files)
    files = json.dumps(files)
    return files




@stable_diffusion_fp16.api(input=Image(),output=Image(mime_type="image/png"))
def background_remover(input_image):
    #print(parsed_json)
    image = background_remover_runner.background_remover.run(input_image)
    return image


