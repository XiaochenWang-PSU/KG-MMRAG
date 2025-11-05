import json 
import random
import os, io,  base64, glob, random
from PIL import Image
from utils import *

# Transform path to base64 for Open API prompt
def img_to_data_url(path):
    """Load image, (optionally) downscale, and return data URL for OpenAI vision input."""
    with Image.open(path) as im:
        # (Optional) downscale very large images to save tokens:
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im = im.convert("RGB")
        im.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# Build Prompt
def build_multimodal_input_for_sample(sample, entity2text):
    """
    Returns a Responses API 'content' list mixing text and (optional) images,
    following mode semantics:
      mode 0: (T1, T2) -> (I1, ?)
      mode 1: (I1, I2) -> (T1, ?)
      mode 2: (I1, T1) -> (I2, ?)
    """
    head, tail = sample["example"][0], sample["example"][1]
    question     = sample["question"]
    mode       = int(sample["mode"])
    content = []

    # Systematic header
    description = (
        "You are solving a knowledge-graph analogy with one exemplar and one question.\n"
        "Interpret (T) as text-only, (I) as image-only.\n"
        "You have to infer the relation hinted by the exemplar to get the relation between question and answer"
    )
    content.append({"type": "input_text", "text": description})

    if mode == 0:
        # (T1, T2) -> (I1, ?)
        head_txt = entity2text[head]
        tail_txt = entity2text[tail]
        content.append({"type": "input_text", "text": f"Exemplar (T1, T2): head = {head_txt} and tail = {tail_txt}"})
        content.append({"type": "input_text", "text": "Question (I1, ?): head = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(question, "images_subset_inference"))})
        content.append({"type": "input_text", "text": f" and tail = ?"})
    elif mode == 1:
        # (I1, I2) -> (T1, ?)
        question_txt = entity2text[question]
        content.append({"type": "input_text", "text": f"Exemplar (T1, T2): head = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(head, "images_subset_inference"))})
        content.append({"type": "input_text", "text": f" and tail = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(tail, "images_subset_inference"))})
        content.append({"type": "input_text", "text": f"Question (T1, ?): head= {question_txt} and tail = ?"})
    else:
        # (I1, T1) -> (I2, ?)
        tail_txt = entity2text[tail]
        content.append({"type": "input_text", "text": f"Exemplar (T1, T2): head = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(head, "images_subset_inference"))})
        content.append({"type": "input_text", "text": f" and tail = {tail_txt}"})
        content.append({"type": "input_text", "text": "Question (I2, ?): head = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(question, "images_subset_inference"))})
        content.append({"type": "input_text", "text": f" and  tail = ?"})

    return content

# Build Prompt for Retrieved Item
def build_rag_prompt(retrieved_items, entity2text, relation2text):
    rag_prompt = []
    rag_prompt.append({"type": "input_text", "text": f"You can use the following knowledge-graph triples as evidence to solve the following question"})
    for (i, item) in enumerate(retrieved_items):
        head, relation, tail = item["item"]
        head_txt = entity2text[head] if head in entity2text else ""
        tail_txt = entity2text[tail] if tail in entity2text else ""
        relation_txt = relation2text[relation] if relation in relation2text else ""
        rag_prompt.append({"type": "input_text", "text": f"Triplet {i+1}: (head, relation, tail) = ({head_txt}, {relation_txt}, {tail_txt})"})

        if first_jpg_path(head, "images_subset_kg"):
            with Image.open(first_jpg_path(head, "images_subset_kg")) as im:    
                rag_prompt.append({"type": "input_text", "text": f"Image for head of triplet {i+1}"})
                rag_prompt.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(head, "images_subset_kg"))})

        if first_jpg_path(tail, "images_subset_kg"):
            with Image.open(first_jpg_path(tail, "images_subset_kg")) as im:    
                rag_prompt.append({"type": "input_text", "text": f"Image for tail of triplet {i+1}"})
                rag_prompt.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(tail, "images_subset_kg"))})

    return rag_prompt
