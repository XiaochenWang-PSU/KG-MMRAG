import os
import re
import json
import argparse
import random
from tqdm import tqdm
from utils import *
from datasets import load_dataset

import openai
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def load_data(args):
    train = load_dataset('derek-thomas/ScienceQA', split='train') # choose the test set
    test = load_dataset('derek-thomas/ScienceQA', split='test')
    # val = load_dataset('derek-thomas/ScienceQA', split='validation')

    test = test.select(range(args.test_number)) if args.test_number > 0 else qids

    # pick up shot examples from the training set
    shots = train.shuffle(args.seed).select(range(args.shot_number)) # random sample
    
    return test, shots


def get_gpt_result(prompt, base64_images, args):
    content = [{ "type": "input_text", "text": f"{prompt}" }]
    for base64_image in base64_images:
        content.append({"type": "input_image","image_url": f"data:image/jpeg;base64,{base64_image}"})

    response = client.responses.create(
        model=args.model,
        input=[
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,  # note the new name
        top_p=args.top_p,
    )
    output = response.output_text.strip()

    # extract the answer
    pattern = re.compile(r'The answer is ([A-Z]).')
    res = pattern.findall(output)
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"

    return answer, output


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


def get_result_file(args):
    result_file = "{}/{}/{}_{}_{}_{}_seed_{}_multimodal.json".format(args.output_root, args.model, args.label, args.test_split,
                                                          args.prompt_format, args.shot_number, args.seed)

    return result_file


def save_results(result_file, acc, correct, count, args, results, outputs):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['args'] = vars(args)
    data['results'] = results
    data['outputs'] = outputs

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/scienceqa')
    parser.add_argument('--output_root', type=str, default='results')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_number', type=int, default=10, help='GPT-3 is expensive. -1 for whole val/test set')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
                        ],
                        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=5, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    test, shots = load_data(args)  # problems, test question ids, shot example ids

    result_file = get_result_file(args)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(result_file))
        acc = check_point['acc']
        correct = check_point['correct']
        results = check_point['results']
        outputs = check_point['outputs']
        print(f"{len(results)}/{len(test)}, correct: {correct}, acc: {round(acc, 2)}%")
    else:
        correct = 0
        results = {}
        outputs = {}

    for qid, problem in enumerate(test):
        if qid in results:
            continue

        choices = problem["choices"]
        answer = problem["answer"]  # 0, 1, ..., 4
        label = args.options[answer]  # 'A', ..., 'E'

        # generate prompt
        prompt, base64_images = build_prompt_multimodal(shots, problem, args)
        
        # generate prediction
        prediction, output = get_gpt_result(prompt, base64_images, args)  # 'A', ..., 'E'
        pred_idx = get_pred_idx(prediction, choices, args.options)  # 0, 1, ..., 4

        results[qid] = pred_idx
        outputs[qid] = output
        if pred_idx == answer:
            correct += 1

        acc = correct / len(results) * 100

        if args.debug or qid < 3:
            print("##################################")
            print(prompt, "\n")
            print("# labeled answer:", label)
            print("# predicted answer:", prediction)
            print("# predicted index:", pred_idx)
            print("# predicted output:", output)

        if (qid + 1) % args.save_every == 0 or (qid + 1) == len(test):
            print(f"{len(results)}/{len(test)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
            save_results(result_file, acc, correct, qid + 1, args, results, outputs)