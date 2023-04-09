import torch
import bentoml
from rembg import remove


class ImageRemoverRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def background_remover(self, image):
        #print("hello")

        image = remove(image)#png 출력
        
        return image