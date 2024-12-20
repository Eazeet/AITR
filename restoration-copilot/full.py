from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uuid
import shutil
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
import os
from helper_functions import sample_frames

# Initialize LLaVA model and processor
multimodal_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-interleave-qwen-0.5b-hf", torch_dtype=torch.float16
).to("cuda")
processor = AutoProcessor.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")

app = FastAPI()

# Temporary directory for storing uploaded files
TEMP_DIR = "./temp_uploads/"
os.makedirs(TEMP_DIR, exist_ok=True)

# Define input schema
class InputRequest(BaseModel):
    text: Optional[str] = None
    files: Optional[List[str]] = None  # List of file paths or upload names

def save_temp_file(upload_file: UploadFile) -> str:
    """Save the uploaded file temporarily and return its path."""
    file_name = f"{uuid.uuid4()}_{upload_file.filename}"
    file_path = os.path.join(TEMP_DIR, file_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

@app.post("/multimodal_inference/")
async def multimodal_inference(text: str, files: List[UploadFile]) -> str:
    """
    Performs multimodal inference using LLaVA with input text and either images or videos.
    
    Args:
        text (str): The input query text.
        files (List[UploadFile]): A list of uploaded image or video files.
        
    Returns:
        str: The generated output based on the text and visual input.
    """
    if not text or not files:
        raise HTTPException(status_code=400, detail="Text and at least one media file are required.")
    
    image_files = []
    for file in files:
        file_path = save_temp_file(file)
        image_files.append(file_path)

    video_extensions = ("avi", "mp4", "mov", "mkv", "flv", "wmv", "mjpeg")
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    if len(image_files) == 1:
        if image_files[0].lower().endswith(video_extensions):
            # Process video and sample frames
            image = sample_frames(image_files[0], 12)
            image_tokens = "<image>" * 13
            prompt = f"<|im_start|>user {image_tokens}\n{text}<|im_end|><|im_start|>assistant"
        elif image_files[0].lower().endswith(image_extensions):
            image = Image.open(image_files[0]).convert("RGB")
            prompt = f"<|im_start|>user <image> The response to the query must be in very great detail \n{text}<|im_end|><|im_start|>assistant"
    elif len(image_files) > 1:
        # Process multiple images/videos
        image_list = []
        for img in image_files:
            if img.lower().endswith(image_extensions):
                img = Image.open(img).convert("RGB")
                image_list.append(img)
            elif img.lower().endswith(video_extensions):
                frames = sample_frames(img, 6)
                image_list.extend(frames)

        toks = "<image>" * len(image_list)
        prompt = f"<|im_start|>user {toks}\n{text}<|im_end|><|im_start|>assistant"
        image = image_list

    # Run the multimodal inference
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
    output = multimodal_model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generated_output = processor.decode(output[0][2:], skip_special_tokens=True)

    # Clean up the temporary files
    for file_path in image_files:
        os.remove(file_path)

    return generated_output
