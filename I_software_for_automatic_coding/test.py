from transformers import BertTokenizerFast, BertForSequenceClassification

import json

from data_split import training_data
import argparse
import pdb
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint')
parser.add_argument('--datapath')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_path = args.datapath

train_set, test_set = training_data(data_path)

class_names=['traditional', 'non-traditional']

model_name = 'bert-base-uncased'
epochs = 2
max_length = 512

def get_splits(txt) :
    txt_ = txt.split()
    num_splits = len(txt_) // 512   + 1

    splits = []
    for i in range(num_splits) :
        splits.append(" ".join(x for x in txt_[512*i:512*(i+1)]))

    return splits

#tokenizer = BertTokenizerFast.from_pretrained('bert_v4-bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)

model = BertForSequenceClassification.from_pretrained('results/checkpoint-'+ str(args.checkpoint)).to(device)

model.eval()

def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return class_names[probs.argmax()], probs



trad = 0
non_trad = 0

correct_trad = 0
correct_non_trad = 0

result = {}
i = 0
my = "|Reference|Coding|\n|:-----|:------|"
for j in test_set :
    i += 1
    result[i] = {}
    txt_sample = j['text']
    coding_sample = j['coding']

    txt_splits = get_splits(txt_sample)
    


    if coding_sample == 'traditional' :
        coding = 0
    else :
        coding = 1

    pred_class = 0
    result[i]['probs'] = {}
    temp = 0 
    for txt in txt_splits :
        
        pred, probs = get_prediction(txt)
        result[i]['probs'][temp] = {}
        result[i]['probs'][temp]['0'] = str(probs.cpu().detach().numpy()[0][0])
        result[i]['probs'][temp]['1'] = str(probs.cpu().detach().numpy()[0][1])

        if pred != 'traditional' :
            pred_class = 1
       
        temp += 1

    result[i]['coding'] = coding
    result[i]['pred_class'] = pred_class
    result[i]['reference'] = j["reference"]
    my_pred = "traditional" if pred_class == 0 else "non-traditional"
    my = my + f"|{j['reference']}|{my_pred}|\n"
    if coding == 0 :
        trad += 1
        if coding == pred_class :
            correct_trad += 1
    else :
        non_trad += 1
        if coding == pred_class :
            correct_non_trad += 1

print ('Traditional correct : ' + str(correct_trad) + '   /   ' + str(trad))
print ('Non traditional correct : ' + str(correct_non_trad) + '   /   ' + str(non_trad))

with open('results.json', 'w+') as f :
    json.dump(result, f, indent=4, ensure_ascii=False)

    
with open("J_Autocoding_results.md", "w")  as f:
    f.write(my) 