import json 
import random
import os, io,  base64, glob, random
from PIL import Image
from openai import OpenAI
from utils import *

client = OpenAI()

# Transform path to base64 for Open API prompt
def img_to_data_url(path):
    """Load image, (optionally) downscale, and return data URL for OpenAI vision input."""
    with Image.open(path) as im:
        # (Optional) downscale very large images to save tokens:
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
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

# Get Metric
def get_metrics(rankings, answers):
    hits1 = 0
    hits3 = 0
    hits5 = 0
    hits10 = 0
    mrr = 0
   
    for i in range(len(rankings)):
        if answers[i] in rankings[i]:
            rank = rankings[i].index(answers[i]) + 1
            hits1 += int(rank <= 1)
            hits3 += int(rank <= 3)
            hits5 += int(rank <= 5)
            hits10 += int(rank <= 10)
            mrr += 1.0/rank
    
    return hits1/len(rankings), hits3/len(rankings), hits5/len(rankings), hits10/len(rankings), mrr/len(rankings)

def rank_candidates_with_gpt(content, candidates):
    """Ask GPT to return a strict ranking of candidate IDs (most->least likely)."""
    rules = f"""
You solve analogies with a single exemplar and a question in one of 3 modes:
- mode 0: (T1, T2) -> (I1, ?)
- mode 1: (I1, I2) -> (T1, ?)
- mode 2: (I1, T1) -> (I2, ?)

CANDIDATE ENTITIES (choose ALL and rank them):
{json.dumps(candidates, ensure_ascii=False)}

Task:
- Rank ALL candidates from most likely to least likely tail for the QUESTION.
- Use the exemplar and mode semantics above to infer the relation.
- Return STRICT JSON ONLY:
{{
  "ranking": ["entity_most_likely", "entity_2", ..., "entity_least_likely"]
}}
- Include each candidate exactly once; no extra items.
    """.strip()

    schema = {
        "format":{
            "type": "json_schema",
            "name": "analogy_ranking",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ranking"],
                "properties": {
                    "ranking": {
                        "type": "array",
                        "items": {"type": "string", "enum": candidates},
                        "minItems": 10,
                        "maxItems": 15,
                    }
                },
                "required": ["ranking"],
            },
            "strict": True,
        }
    }

    resp = client.responses.create(
        model="gpt-4o-2024-08-06",
        input=[
            
            {
                "role": "system",
                "content": rules,
            },
            {"role": "user", "content": content},
        
        ],
        temperature=0,
        max_output_tokens=512,
        text = schema
    )
    return json.loads(resp.output_text.strip())


# Read test 
with open("dataset/MARS/test.json", "r", encoding="utf-8") as f:
    lines = f.readlines()
    test_samples = [json.loads(line) for line in lines]
    

entity2text = read_txt("dataset/MarKG/entity2text.txt")
candidates = []

for sample in test_samples[:100]:
    candidates.append(entity2text[sample["answer"]])

test_samples = random.sample(test_samples, 5)

rankings = []
answers = []

for sample in test_samples:
    content = build_multimodal_input_for_sample(sample, entity2text)
    rankings.append(rank_candidates_with_gpt(content, candidates)["ranking"])
    answers.append(entity2text[sample["answer"]])
    print(answers[-1], rankings[-1])

metrics = get_metrics(rankings, answers)
print("Hits@1", metrics[0])
print("Hits@3", metrics[1])
print("Hits@5", metrics[2])
print("Hits@10", metrics[3])
print("MRR", metrics[4])