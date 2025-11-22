from PIL import Image
import os, io,  base64, glob, random
from matplotlib import pyplot as plt

# Transform path to base64 for Open API prompt
def path_to_data_url(path):
    """Load image, (optionally) downscale, and return data URL for OpenAI vision input."""
    with Image.open(path) as im:
        # (Optional) downscale very large images to save tokens:
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# Build Prompt
def build_multimodal_input_for_sample(question):

    img_url = path_to_data_url("new_dataset_release/new_dataset_release/images/"+question['img_file'])

    return [
        {
            "role": "system",
            "content": (
                "You are a fact-based visual question answering model."
                "You MUST answer with a short from the fact, and nothing else."
                "No punctuation, no explanation."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": img_url,
                },
                {
                    "type": "input_text",
                    "text": f"Question: {question['question']}",
                },
                {
                    "type": "input_text",
                    "text": f"Fact: {question['fact_surface'].replace('[','').replace(']','')}",
                }
            ],
        },
    ]
