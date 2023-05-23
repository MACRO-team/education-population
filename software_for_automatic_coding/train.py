from data_split import training_data_sentences
import os
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments


os.environ["WANDB_DISABLED"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = 'dataset_master_with_sections_and_sentences.json'

train_set, test_set = training_data_sentences(data_path)

class_names=['traditional', 'non-traditional']

model_name = 'bert-base-uncased'

tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(class_names)).to(device)

epochs = 2
max_length = 512

def get_splits(txt) :
    txt_ = txt.split()
    num_splits = len(txt_) // 512   + 1

    splits = []
    for i in range(num_splits) :
        splits.append(" ".join(x for x in txt_[512*i:512*(i+1)]))

    return splits



train_texts = []
train_labels = []
for j in train_set :
    txt_sample = j['text']
    coding_sample = j['coding']

    txt_splits = get_splits(txt_sample)

    if coding_sample == 'traditional' :
        coding = 0
    else :
        coding = 1

    for txt in txt_splits :
        train_texts.append(txt)
        train_labels.append(coding)

test_texts = []
test_labels = []
for j in test_set :
    txt_sample = j['text']
    coding_sample = j['coding']

    txt_splits = get_splits(txt_sample)

    if coding_sample == 'traditional' :
        coding = 0
    else :
        coding = 1

    for txt in txt_splits :
        test_texts.append(txt)
        test_labels.append(coding)


train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)


class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, test_labels)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)


def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.25])).to(device)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=25,               # log & save weights each logging_steps
    save_steps=25,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

trainer = CustomTrainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

# train the model
trainer.train()

# saving the fine tuned model & tokenizer
model_path = "bert_v4-bert-base-uncased"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)



