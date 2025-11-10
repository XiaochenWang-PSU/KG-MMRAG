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

# Transform image to base64 for Open API prompt
def img_to_data_url(img):
    im = img.copy()
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# Build Prompt
def build_multimodal_input_for_sample(sample):

    if "image_path" in sample:
        img_url = path_to_data_url(sample["image_path"])
    else:
        img_url = img_to_data_url(sample["image"])

    return [
        {
            "role": "system",
            "content": (
                "You are a medical visual question answering model. "
                "You MUST answer using exactly one character: "
                "'1' if the correct answer is yes/true, or '0' if the correct answer is no/false. "
                "No words, no punctuation, no explanation."
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
                    "text": f"Question: {sample['question']}\nAnswer with 1 for yes/true or 0 for no/false.",
                },
            ],
        },
    ]