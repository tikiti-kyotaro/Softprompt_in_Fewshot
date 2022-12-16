import argparse
import logging

import numpy as np
import torch
import json

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    BertForMaskedLM, BertModel,
    BertTokenizer, BertTokenizerFast, AutoConfig,
    set_seed,
    GPT2LMHeadModelAdapter,
)
import sys, os
from train_control import PrefixTuning, PrefixEmbTuning
import pickle


MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}


model_name_or_path = "gpt2-large"
cache_dir = "/home/kyotaro/prompt-order/PrefixTuning/cache/"
model_type = "gpt2"
tuning_mode = 'prefixtune'
task_mode = 'classify-sentiment'
objective_mode = 2


model_class, tokenizer_class = MODEL_CLASSES[model_type]

config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
print(config)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
config._my_arg_tune_mode = tuning_mode
config._my_arg_task_mode = task_mode
config._objective_mode = objective_mode
gpt2 = model_class.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

# b = list(model.parameters())

# この input_text を埋め込んだものが vector に入る
input_text = "This is Halloween ! "
# input_text = "Today is sunny .  So I want to go the park .  It was "

text_index = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
print(f'text_index : {text_index}')
# vector がうまくいってない？
# vector = gpt2.transformer.wte(text_index)
# print(f'vector : {vector}')

with open("/home/kyotaro/prompt-order/PrefixTuning/prompt.pkl", "rb") as in_:
    prompt = pickle.load(in_)

# gpt2.to("cuda:0")
# text_index.to("cuda:0")


output_sequences = gpt2.generate(
    input_ids=text_index,
    emb_match=None,
    control_code=None,
    max_length=20,
    min_length=5,
    temperature=1.0,
    top_k=0,
    top_p=0.5,
    bad_words_ids=[[628], [198]] if True else None,
    repetition_penalty=1.0,
    do_sample=False,
    num_return_sequences=1,
) # max_length だけ適当

print(f'output_sequences : {output_sequences}')

for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
    # args.stop_token = tokenizer.eos_token
    generated_sequence = generated_sequence.tolist()
    # Decode text
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    print(text)
    text_output = text[len(tokenizer.decode(vector[0][0], clean_up_tokenization_spaces=True)):]
    # print(text_output)