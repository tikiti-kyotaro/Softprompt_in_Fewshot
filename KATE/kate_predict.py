import pickle
import sys
import random
import hard_predict as hard
# import soft_predict as soft
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel


test_file = sys.argv[1]
dict_file = sys.argv[2]
knn_file = sys.argv[3]
contexts_num = int(sys.argv[4])
mode = sys.argv[5]

def load_train_dict(dict_file):
    with open(dict_file, "rb") as dict_:
        train_dict = pickle.load(dict_)
    return train_dict

def load_knn_list(knn_file):
    with open(knn_file, "rb") as knn:
        knn_list = pickle.load(knn)
    return knn_list

def nearest_contexts_list(contexts_num, test_id, train_dict, knn_list):
    contexts_list = list()
    for k in range(len(knn_list[test_id])):
        if k < contexts_num:
            context_id = knn_list[test_id][k]
            # print(train_dict[context_id])
            contexts_list.append(train_dict[context_id])
    return contexts_list

def model_input_for_no_template(contexts_list, test_sentence):
    p = ""
    for context in contexts_list:
        p += context + " "
    model_input = p + test_sentence
    return model_input

def model_input_for_itis(contexts_list, test_sentence):
    p = ""
    for context in contexts_list:
        context_sentence, context_label = " ".join(context.split(" ")[:-1]), context.split(" ")[-1]
        # print(context_sentence)
        # print(context_label)
        context_formated = f'{context_sentence}\nit is {context_label}\n'
        p += context_formated
    model_input = p + f'{test_sentence}\nit is '
    # print(model_input)
    return model_input

def model_input_for_review_sentiment(contexts_list, test_sentence):
    p = ""
    for context in contexts_list:
        context_sentence, context_label = " ".join(context.split(" ")[:-1]), context.split(" ")[-1]
        # print(context_sentence)
        # print(context_label)
        context_formated = f'Review : {context_sentence}\nSentiment : {context_label}\n'
        p += context_formated
    model_input = p + f'Review : {test_sentence}\nSentiment : '
    return model_input

def make_model_input(test_file, dict_file, knn_file):
    train_dict = load_train_dict(dict_file)
    knn_list = load_knn_list(knn_file)

    with open(test_file, "r") as test:
        for test_id, line in tqdm(enumerate(test)):
            # if test_id < 2:  # めんどくさいからここだけ
            line = line.strip()
            contexts_list = nearest_contexts_list(contexts_num=contexts_num, test_id=test_id, train_dict=train_dict, knn_list=knn_list)   
            if mode == "no_template":         
                model_input = model_input_for_no_template(contexts_list=contexts_list, test_sentence=line)
            elif mode == "itis":
                model_input = model_input_for_itis(contexts_list=contexts_list, test_sentence=line)
            elif mode == "review-sentiment":
                model_input = model_input_for_review_sentiment(contexts_list=contexts_list, test_sentence=line)
            # print(model_input)
            hard.main_hard(model_input=model_input, tokenizer=tokenizer, model=model, device=device)
    return model_input

def main():
    model_input = make_model_input(test_file=test_file, dict_file=dict_file, knn_file=knn_file)
    hard.main_hard(model_input=model_input, tokenizer=tokenizer, model=model, device=device)




if __name__ == "__main__":
    device = "cuda:1"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    random.seed(0)

    main()