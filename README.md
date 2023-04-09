# dotting-diffusion



## API example
```


1. txt2img

request JSON

% curl -X POST -H "content-type: application/json" \
    --data '{"prompt": "astronaut riding a brown horse"}' \
    http://localhost:3000/txt2img

ip:3000/txt2img/

output : image/png

```

```
2.  background remover

request MultiPart

% curl -H "Content-Type: multipart/form-data" \
       -F 'fileobj=@test.jpg;type=image/jpeg' \
       http://localhost:3000/background_remover


ip:3000/background_remover/

output : image/png



```

request.ipynb 참조


## Plan


1. Multiple sample geneation in txt2img
2. Image saving in storage bucket and sending url to client
3. Moving to serverless inference
4. More features using controlnet
5. Data collection and fine-tuning

