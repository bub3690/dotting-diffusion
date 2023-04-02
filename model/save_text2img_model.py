#write code to load stable diffusion model using diffuser
#write code to save model to file

import torch
from diffusers import DiffusionPipeline
import bentoml


def load_diffusion_model_save_it_to_bento(model_path: str,model_name: str):
    
    # load model
    pipeline = DiffusionPipeline.from_pretrained(model_path,torch_dtype=torch.float16,is_accelerate_available=False)
    # save model to bentoml
    bento_model = bentoml.diffusers.save_model(model_name, pipeline )
    print("Bento model tag : {}".format(bento_model))
    


if __name__ =="__main__":
    load_diffusion_model_save_it_to_bento("PublicPrompts/All-In-One-Pixel-Model","txt2img_pixel_art_diffusion")

