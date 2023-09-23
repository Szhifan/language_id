from transformers import pipeline
import os 
from nltk.tokenize import sent_tokenize
import pickle 
import random
import tqdm 
from ngram_lm.model_implementation import * 
from mbert_lang_id.utils import test_model
from mbert_lang_id import models,load_data
random.seed(42)
def roberta(sents:list):
    pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    langs = [i["label"] for i in pipe(sents)]
    return langs 
def n_gram(sents:list):
    model_dir = "ngram_lm/models"
    model_list = os.listdir(model_dir)  
    langs = []
    for sent in tqdm.tqdm(sents):
        lang = ""
        p = 1000000
        for model_path in model_list:
            if model_path.endswith("dict"):
                model = pickle.load(open(model_dir+"/"+model_path,"rb"))
                p_lang = perplexity(sent,model)
                if p > p_lang:
                    p = p_lang
                    lang = model_path[:2]
        langs.append(lang)
    return langs
def mbert(sents:list):

    dts = load_data.BertDataset("mbert_lang_id/lang_id_data/lang_id_36.csv",labels_only=True)
    model = models.BertSeqClassifier(len(dts.labels2id))
    encoding = dts.tokenizer(sents,return_tensors="pt",padding=True,max_length=100,truncation=True)
    langs = test_model("mbert_lang_id/results/lang36/checkpoint_best.pt",dts,model,encoding)
    return langs 
def get_model(model):
    if model == "roberta":
        return roberta 
    elif model== "ngram":
        return n_gram
    elif model == "mbert":
        return mbert 
def random_sample(items:list,n:int):
    """
    randomly sample n items from a list, each item in the list can only be chosen once.
    """
    length = len(items)
    if n>=length:
        return items 
    chosen_items = []
    for _ in range(n):
        item = random.choice(items)
        items.remove(item)
        chosen_items.append(item)
    return chosen_items
def detect_language(path:str,model="mbert",sample_rate=0.5):
    """
    path: path to the document 
    model: choose a model. available choices: roberta, ngram, mbert
    sample_rate: randomly sample x% of the sentences in the document. 
    This function takes a text file as input and return a list of language(s) the document is 
    written in. 
    """
    model = get_model(model)
    text = open(path,"r").read()
    text = re.sub(r"\n","",text)
    text_list = sent_tokenize(text)
    text_list = random_sample(text_list,int(len(text_list)*sample_rate))
    langs = model(text_list)
    lang_freq = collections.Counter(langs)
    return lang_freq.most_common(1)[0][0]
if __name__ == "__main__":
    path = "test/mix-de-zh.txt"
    print(detect_language(path))
