import argparse
from prompt_builder import *
from openai import OpenAI
import json 

client = OpenAI()

def get_gpt_result(prompt):
    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0,
        max_output_tokens=512,
        timeout=60,
    )
    return resp.output_text.strip()

def compute_accuracy(outputs, answers):
    count_correct = 0
    
    for i in range(len(outputs)):
        if outputs[i].replace("a ","").replace("an ","").replace("the ","") == \
            answers[i].replace("a ","").replace("an ","").replace("the ",""):
            count_correct += 1
    
    return {"ACC": 100*count_correct/len(outputs)}

if __name__ == "__main__":
    with open("new_dataset_release/new_dataset_release/all_qs_dict_release.json", 'r', encoding='utf-8') as file:
        questions = json.load(file)
        
    outputs = []
    answers = []

    for question_id in questions:
        prompt = build_multimodal_input_for_sample(questions[question_id])
        outputs.append(get_gpt_result(prompt).lower().strip())
        answers.append(questions[question_id]['answer'].lower().strip())
        
    print(compute_accuracy(outputs, answers))