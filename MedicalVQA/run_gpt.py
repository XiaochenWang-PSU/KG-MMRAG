import argparse
from data import MedicalVQADataset
from prompt_builder import *
from openai import OpenAI
from utils import *

client = OpenAI()

def get_gpt_result(prompt):
    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0,
        max_output_tokens=512,
    )
    return resp.output_text.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # None
    parser.add_argument('--retriever', type=str, default=None, help='Retriever') 
    # 'slake', 'vqa_rad', 'pathvqa'
    parser.add_argument("--dataset", type=str, default='slake')

    args = parser.parse_args()

    vqa_data = MedicalVQADataset(args.dataset, split="test")

    outputs = []
    answers = []

    for sample in vqa_data.samples[:10]:
        prompt = build_multimodal_input_for_sample(sample)
        outputs.append(int(get_gpt_result(prompt)))
        answers.append(int(sample["answer"]))

    print(compute_metrics(answers, outputs))