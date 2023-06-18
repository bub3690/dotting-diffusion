set MODEL_NAME="runwayml/stable-diffusion-v1-5"
set INSTANCE_DIR="D:/project/dotting_ai/data/CHAR"
set OUTPUT_DIR="D:/project/dotting_ai/data/dreambooth/spritesheet"

accelerate launch train_dreambooth.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --instance_data_dir="D:/project/dotting_ai/data/CHAR" --output_dir="D:/project/dotting_ai/data/dreambooth/spritesheet" --instance_prompt="DonaldTrump, spritesheet" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=400

