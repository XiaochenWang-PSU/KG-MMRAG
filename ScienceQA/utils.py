import base64
from io import BytesIO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch
import os, json, random, re, torch, base64, io
from PIL import Image

def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context

def get_context_multimodal(problem, use_caption):
    hint = problem['hint']
    print(problem['image'])
    base64_image = pil2base64(problem['image']) if problem['image'] else None
    if hint == "":
        hint = "N/A"
    return hint, base64_image

def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text


def build_prompt(problems, shot_qids, test_qid, args):

    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(args.prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    examples.append(test_example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input



class QwenLocal:
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None
        )

    def _decode_data_url(self, data_url):
        if not data_url.startswith("data:"):
            raise ValueError("Expect data URL")
        header, b64data = data_url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")

    def generate(self, segments, max_new_tokens=512, temperature=0.2, top_p=0.95):
        content, images = [], []
        for seg in segments:
            if seg["type"] in ("input_text", "text"):
                content.append({"type": "text", "text": seg["text"]})
            elif seg["type"] in ("input_image", "image"):
                img = self._decode_data_url(seg["image_url"])
                content.append({"type": "image"})
                images.append(img)
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=(temperature > 0),
                                      temperature=temperature, top_p=top_p)
        return self.processor.decode(out[0], skip_special_tokens=True)
        
        
def build_prompt_multimodal(shots, problem, args):

    examples = []
    base64_images = []

    # n-shot training examples
    for shot in shots:
        question = get_question_text(shot)
        hint, base64_image = get_context_multimodal(shot, args.use_caption)
        if base64_image:
            base64_images.append(base64_image)
        choice = get_choice_text(shot, args.options)
        answer = get_answer(shot, args.options)
        lecture = get_lecture_text(shot)
        solution = get_solution_text(shot)

        train_example = create_one_example(args.prompt_format,
                                           question,
                                           hint,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        examples.append(train_example)

    # test example
    question = get_question_text(problem)
    hint, base64_image = get_context_multimodal(problem, args.use_caption)
    if base64_image:
        base64_images.append(base64_image)
    
    choice = get_choice_text(problem, args.options)
    answer = get_answer(problem, args.options)
    lecture = get_lecture_text(problem)
    solution = get_solution_text(problem)

    test_example = create_one_example(args.prompt_format,
                                      question,
                                      hint,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    examples.append(test_example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)
    return prompt_input, base64_images

def build_segments_multimodal(shots, problem, args):
    """
    Returns a list of interleaved content items for the Responses API:
      [{"type":"input_text","text":...},
       {"type":"input_image","image_url":"data:image/jpeg;base64,..."},
       ...]
    Order is preserved (text, image, text, image, ...).
    """

    segments = []

    def add_example(one, test_example: bool):
        question = get_question_text(one)
        hint, base64_image = get_context_multimodal(one, args.use_caption)
        choice = get_choice_text(one, args.options)
        answer = get_answer(one, args.options)
        lecture = get_lecture_text(one)
        solution = get_solution_text(one)

        # Build the full example text block exactly as before
        example_text = create_one_example(
            args.prompt_format, question, hint, choice, answer, lecture, solution,
            test_example=test_example
        )
        # 1) push the text
        segments.append({"type": "input_text", "text": example_text})

        # 2) then, if there's an image for this example, push it right after
        if base64_image:
            segments.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            })

    # n-shot examples
    for shot in shots:
        add_example(shot, test_example=False)
        add_example(shot, test_example=False)

    # test example
    add_example(problem, test_example=True)

    return segments


def pil2base64(img):
    buf = BytesIO()
    fmt = img.format
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64