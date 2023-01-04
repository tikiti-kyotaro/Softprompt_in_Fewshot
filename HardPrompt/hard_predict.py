import torch
import torch.nn.functional as F
import sys

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import random


device = "cuda:0"
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
# model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
# model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model = model.to(device)
random.seed(0)

def get_logprobs(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(device)
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    input_ids = input_ids.to(device)
    output_ids = output_ids.to(device)
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logits = logits.to(device)
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    logprobs = logprobs.to(device)
    return logprobs

def COPA_eval(prompt, alternative1, alternative2):
    lprob1 = get_logprobs(prompt + "\n" + alternative1).sum()
    lprob2 = get_logprobs(prompt + "\n" + alternative2).sum()

    # print(f'{alternative1}:{lprob1}')
    # print(f'{alternative2}:{lprob2}')
    return 0 if lprob1 > lprob2 else 1

def output_result(result, alternative1, alternative2):
    if result:
        print(alternative2)
    else:
        print(alternative1)


# 0ならchoice1, 1ならchoice2
choice1 = "positive"
choice2 = "negative"

prompt_sample = sys.argv[1]
input_file = sys.argv[2]
mode = sys.argv[3]

def predict_no_template(prompt_sample, input_file):
    p = ""
    with open(prompt_sample, "r") as prompt:
        for line in prompt:
            p = p + line

    with open(input_file, "r") as input_:
        for i, line in tqdm(enumerate(input_)):
            line = line.strip()
            test_input = f'{line} '
            model_input = p + test_input
            # print(model_input)
            predict = COPA_eval(model_input, choice1, choice2)
            output_result(predict, choice1, choice2)

def predict_itis(prompt_sample, input_file):
    p = ""
    with open(prompt_sample, "r") as prompt:
        for line in prompt:
            p = p + line

    with open(input_file, "r") as input_:
        for i, line in tqdm(enumerate(input_)):
            line = line.strip()
            test_input = f'{line}\nit is '
            model_input = p + test_input
            # print(model_input)
            predict = COPA_eval(model_input, choice1, choice2)
            output_result(predict, choice1, choice2)

def predict_review_sentiment(prompt_sample, input_file):
    p = ""
    with open(prompt_sample, "r") as prompt:
        for line in prompt:
            p = p + line

    with open(input_file, "r") as input_:
        for i, line in tqdm(enumerate(input_)):
            line = line.strip()
            test_input = f'Review : {line}\nSentiment : '
            model_input = p + test_input
            # print(model_input)
            predict = COPA_eval(model_input, choice1, choice2)
            output_result(predict, choice1, choice2)

def main(prompt_sample, input_file, mode):
    if mode == "no_template":
        predict_no_template(prompt_sample, input_file)
    elif mode == "itis":
        predict_itis(prompt_sample, input_file)
    elif mode == "review-sentiment":
        predict_review_sentiment(prompt_sample, input_file)

# prompt_sample = "/home/kyotaro/Hard-prompt/contexts/itis/seed_1/context_16.txt"
# input_file = "/home/kyotaro/Hard-prompt/base_data/test/test_sentence.txt"
# predict_itis(prompt_sample, input_file)

if __name__ == "__main__":
    main(prompt_sample, input_file, mode)