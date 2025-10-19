import json 
import random
import os, io,  base64, glob, random
from PIL import Image
from openai import OpenAI
import matplotlib.pyplot as plt

# Get first jpg from entity folder
def first_jpg_path(entity_id, base_dir):
    folder = os.path.join(base_dir, entity_id)
    print(folder)
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    print(os.path.join(folder, "*.jpg"))
    return files[0] if files else None

# Transform path to base64 for Open API prompt
def img_to_data_url(path):
    """Load image, (optionally) downscale, and return data URL for OpenAI vision input."""
    with Image.open(path) as im:
        # (Optional) downscale very large images to save tokens:
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    if b64.strip().startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]  # strip 'data:image/png;base64,'
    img_bytes = base64.b64decode(b64)
    decoded = Image.open(io.BytesIO(img_bytes))
    plt.figure()
    plt.axis("off")
    plt.imshow(decoded)
    plt.show()
    return f"data:image/jpeg;base64,{b64}"

print(img_to_data_url(first_jpg_path("Q467", "images_subset_inference")))