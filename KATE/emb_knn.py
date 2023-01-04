import torch
import faiss
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import sys
import numpy as np
import pickle
import time

device = torch.device('cuda:0')

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.to(device)

train_file = sys.argv[1]
test_file = sys.argv[2]
k = int(sys.argv[3])

def get_file_embeds(input_file, tokenizer, model):
    embeds_list = list()
    #Our sentences we like to encode
    with open(input_file, "r") as input_:
        for i, line in enumerate(tqdm(input_)):
            line = line.strip()
            inputs = tokenizer(line, return_tensors="pt")
            inputs.to(device)

            outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            sentence_embeds = last_hidden_states[0][0]
            
            embeds_list.append(sentence_embeds)
    return embeds_list
    
def cal_dist(train_emb_list, test_emb_list):
    for train in train_emb_list:
        for test in test_emb_list:
            print(torch.dist(train,test,p=2))

def check_shape(emb_list):
    for emb in emb_list:
        print(emb.shape)

def tensor_to_numpy(input_tensor):
    return input_tensor.to('cpu').detach().numpy().copy().astype(np.float32)

def list_to_numpy(input_list):
    result_list = list()
    for tens in input_list:
        result_list.append(tensor_to_numpy(tens))
    result_numpy = np.array(result_list)
    return result_numpy

def dump_numpy(input_numpy, output_file):
    with open(output_file, "wb") as out_:
        pickle.dump(input_numpy, out_)

def knn_faiss(train_num, test_num, k):
    d = 768                         # ベクトルの次元(dimension)

    xb = train_num
    xq = test_num

    print("INPUT NUMPY")
    print(xb.shape)
    print(xq.shape)

    # GpuIndexFlatL2オブジェクトを作成
    print("BUILD THE INDEX")
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    index = faiss.GpuIndexFlatL2(res, d, flat_config) 

    #ベクトルを追加(append vectors to the index)
    index.add(xb)
    # print(index.ntotal)  # 1000000

    print("SEARCH KNN")
    # 探索実行(search)
    s = time.time()
    D, I = index.search(xq, k)
    e = time.time()
    print(I)
    print(D)
    print("time: {}".format(e-s))

    return I

def main():
    print("GET EMBEDS")
    print("FAISS FORMAT")
    train_emb_num = list_to_numpy(get_file_embeds(input_file=train_file, tokenizer=tokenizer, model=model))
    test_emb_num = list_to_numpy(get_file_embeds(input_file=test_file, tokenizer=tokenizer, model=model))

    with open("/home/kyotaro/klas/data/knn_list.pkl", "wb") as out_:
        pickle.dump(knn_faiss(train_emb_num, test_emb_num, k), out_)


if __name__ == "__main__":
    main()
