import requests
import json

dat = json.dumps({"prompt": "men"})
res = requests.post("http://localhost:3000/txt2img/", data=dat)