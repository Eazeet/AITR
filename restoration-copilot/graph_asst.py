import os
import shutil
import uuid
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, Request, Response, UploadFile, Form, File
from fastapi.responses import HTMLResponse, JSONResponse

import torch
import numpy as np
import pytz
import openai
from PIL import Image
from dotenv import load_dotenv
# from IPython.display import display, Image
from datasets import Dataset
from typing import List, Annotated, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

from helper_functions import (
    num_tokens_from_string, replace_t_with_space, replace_double_lines_with_one_line, 
    split_into_chapters, analyse_metric_results, escape_quotes, text_wrap, 
    extract_book_quotes_as_documents, get_user_input, sample_frames
)
from some_knowledge_base_functions import create_chapter_summary
from langgraph_helpers import handle_tool_error, create_tool_node_with_fallback, _print_event
from file_encodings import encode_restoration_chapter, encode_preliminary_inspection_chapter, encode_book
from tools import categorize_class, create_objectives, book_lookup, preliminary_chapter_lookup, restoration_chapter_lookup

from state import State
from redis_functions import RedisSaver

from transformers import LlavaForConditionalGeneration, AutoProcessor

os.environ["OPENAI_API_KEY"] = ""
groq_api_key = ""
os.environ["OPENAI_API_KEY"] = ""
app = FastAPI()

# Serve HTML for demo
@app.get("/", response_class=HTMLResponse)
async def demo():
    with open("index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

llm = ChatOpenAI(model="gpt-4o-mini")
multimodal_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-interleave-qwen-0.5b-hf", torch_dtype=torch.float16
).to("cuda")
processor = AutoProcessor.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intelligent building restoration copilot, designed to assist restoration engineers with structural damage restoration tasks. "
            "You will follow a structured process to gather context, categorize damage, and interactively retrieve and provide step-by-step guidance during restoration efforts. "
            "Make sure to provide an output after every tool call, and also ask the user questions"
            "Your workflow should adhere to the following structure:"
            "**Context Gathering from Multimodal Model:** "
            "- Use the multimodal model to gather context from images or videos provided by the user if available. If the input is text-only, analyze the provided text."
            "- Summarize the gathered information before proceeding."
            "\n\n2. **Input Example:** "
            "The input will be in dictionary format, where the 'text' field contains a description of the task or question, and the 'files' field contains optional media files (images or videos). If no media files are provided, only the text will be used for analysis."
            "\n\nIf only text is provided, use the text to infer appropriate tools."
            "\n\n3. **Categorize the Damage Class:** "
            "- After gathering context, categorize the type of damage into one of three categories: Water, Fire, or General."
            "- Provide initial guidance on restoration procedures based on the categorization."
            "\n\n4. **Interactive Preliminary Objectives:** "
            "- Use the `preliminary_chapter_lookup` tool to retrieve relevant steps for preliminary inspection based on the damage type, don't just output the objectives, relate them to what the user is experiencing. " 
            "- this should be interactive, whereby you ask questions from the user based on the current objective to know if the preliminary objective has been fulfilled or not, don't ask too many questions at the same time, when you feel the objectives leading up to restoration have been fulfilled,  you can proceed to the restoration stage"
            "\n\n5. **Restoration Process Retrieval:** "
            "- Use the `restoration_chapter_lookup` tool to retrieve detailed restoration steps."
            "- Make sure the restoration steps go along with whatever damage you have detected, do not present generic restoration procedures."
            "- Present the restoration steps interactively and allow the user to request further clarifications."
            "\n\nThroughout the process, make sure to interactively assist the user, ask for clarifications when needed, and align with the IICRC guidelines."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

part_1_tools = [categorize_class,
                create_objectives,
                preliminary_chapter_lookup,
                restoration_chapter_lookup,
                book_lookup
                ]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

TEMP_DIR = "./temp_uploads/"
os.makedirs(TEMP_DIR, exist_ok=True)

# Define Pydantic model for input request
class InputRequest(BaseModel):
    text: str  # Text input from the user
    files: Optional[List[UploadFile]] = None  # List of image or video files (optional)

# Helper to save temp files
def save_temp_file(upload_file: UploadFile) -> str:
    file_name = f"{uuid.uuid4()}_{upload_file.filename}"
    file_path = os.path.join(TEMP_DIR, file_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path
    
@app.post("/unified_inference/")
async def unified_inference(
    request: Request,
    response: Response,
    text: str = Form(...),  # Text input is now handled using Form
    files: Optional[List[UploadFile]] = File(None)  # Files are uploaded separately using UploadFile and File
):
    """
    Unified inference endpoint that handles both multimodal and textual data.
    If multimodal data (images/videos) is present, it will first route through the multimodal endpoint.
    """
    # Step 1: Check if there are any files (multimodal data)
    if files:
        # Route to multimodal endpoint first
        multimodal_response = await multimodal_inference(text, files)

        # Use the output from multimodal inference to pass into process_restoration (chatbot)
        chatbot_response = await process_restoration(UserQuery(text=multimodal_response), request, response)

        return {
            "multimodal_result": multimodal_response, 
            "chatbot_response": chatbot_response["response"]
        }

    else:
        # If no files, directly route to process_restoration
        chatbot_response = await process_restoration(UserQuery(text=text), request, response)
        return {"chatbot_response": chatbot_response["response"]}


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

class UserQuery(BaseModel):
    text: str  # User's input query or message

@app.post("/process_restoration")
async def process_restoration(question: UserQuery, request: Request, response: Response):
    thread_id = request.headers.get('X-Thread-ID', None)
    if not thread_id:
        thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "user_id": str(uuid.uuid4()),
            "thread_id": thread_id,
        }
    }

    _printed = set()

    # Simulate conversation and checkpointing using RedisSaver
    with RedisSaver.from_conn_info(host="localhost", port=6379, db=0) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        # question = get_user_input()
        # print(f"User: {question}")
        
        events = graph.stream(
        {"messages": ("user", question.text)}, config, stream_mode="values"
        )
        response_text = ""
        latest_checkpoint = checkpointer.get(config)
        latest_checkpoint_tuple = checkpointer.get_tuple(config)
        checkpoint_tuples = list(checkpointer.list(config))
        
        for event in events:
            event_output = _print_event(event, _printed)  # Assuming _print_event handles the event output
            response_text = event_output 

    response.headers["X-Thread-ID"] = thread_id
        
        # Return the assistant's final output
    return {"response": response_text.strip()}