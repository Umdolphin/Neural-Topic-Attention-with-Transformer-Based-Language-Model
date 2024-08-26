## Package load
import re
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import time
import gc
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
stopWords = set(stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer, AlbertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from torchmetrics import F1Score,Precision,Recall

# Set seed
random_seed= 42

# Set Hyperparameter 
window_size = 5
min_count = 5
max_word_num = 500
reduced_vocab_size = 2000
num_classes= 3
topic_hidden_size=64
dropout_keep_proba=0.1
hidden_size = 768
num_topic = 50
embedding_size = 100
threshold = 0.1
learning_rate =  2e-5
num_epochs = 5
max_grad_norm = 1

# Make storage
MRD_ANEM = pd.DataFrame(columns = ['Num_Topic','AUC','Precision','Recall','F1','random_seed','Time'])


np.random.seed(random_seed)
    
## Data load

train_df = pd.read_csv('train_MRD_3label.csv')
train_df = train_df[['text','rating']]

test_df = pd.read_csv('test_MRD_3label.csv')
test_df = test_df[['text','rating']]

train_data_label = train_df['rating'].apply(lambda x: int(x))
train_data_label = train_data_label.tolist()

test_data_label = test_df['rating'].apply(lambda x: int(x))
test_data_label = test_data_label.tolist()
    
device = torch.device("cuda")

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
Albert = AlbertModel.from_pretrained('albert-base-v2')

max_len = 256
    
class CustomDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, Albert_tokenizer, max_len,
                 pad, pair):
        self.sentences = [Albert_tokenizer(text,padding='max_length',return_tensors = "pt",max_length = max_len,truncation=True) for text in dataset['text']]
        self.labels = [np.int32(label) for label in dataset['rating']]
   
    def get_batch_labels(self, i):
        # Fetch a batch of labels
        return np.array(self.labels[i])

    def get_batch_texts(self, i):
        # Fetch a batch of inputs
        return self.sentences[i]    
    
    def __getitem__(self, i):
        batch_texts = self.get_batch_texts(i)
        batch_y = self.get_batch_labels(i)
        return batch_texts, batch_y

    def __len__(self):
        return (len(self.labels))
    
batch_size = 8

train_dataset = CustomDataset(train_df, 0,1, tokenizer, max_len, True, False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_dataset = CustomDataset(test_df, 0,1, tokenizer, max_len, True, False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
class AlbertClassifier(nn.Module):
    def __init__(self,
                 Albert,
                 hidden_size = 768,
                 num_classes=3,
                 dr_rate=None,
                 params=None):
        super(AlbertClassifier, self).__init__()
        self.Albert = Albert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(768, 3)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
            

    def forward(self, token_ids, segment_ids, attention_mask,length):
        pooler = self.Albert(input_ids = token_ids,token_type_ids = segment_ids,attention_mask = attention_mask)[1]

        if self.dr_rate:
            out = self.dropout(pooler)
            
        return self.classifier(out)
    
model = AlbertClassifier(Albert,dr_rate = 0.1).to(device)
    
learning_rate =  2e-5
num_epochs = 5
max_grad_norm = 1
    
# Adamw

t_total = len(train_dataloader) * num_epochs

loss_fn = nn.CrossEntropyLoss()

warmup_ratio = 0.1
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


def accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
    
start = time.time()


train_acc_L = []
test_acc_L = []
train_ce_L = []
test_ce_L = []

train_f1_score = []
test_f1_score = []
train_precision_L = []
test_precision_L = []
train_recall_L = []
test_recall_L = []
    
precision_score = Precision(num_classes = 3, average = 'macro').to(device)
recall_score = Recall(num_classes = 3, average = 'macro').to(device)
f1 = F1Score(num_classes = 3,average='macro').to(device)     

train_acc = 0.0
test_acc = 0.0
train_ce = 0.0
test_ce = 0.0
        
train_f1 = 0
test_f1 = 0
train_precision = 0
test_precision = 0
train_recall = 0
test_recall = 0

# Train


for e in range(num_epochs):
    print("Current epochs: {}".format(e))
    model.train()
    for batch_id, (train_input, train_label) in enumerate((train_dataloader)):
        optimizer.zero_grad()
        token_ids = train_input['input_ids'].squeeze(1).to(device)
        segment_ids = train_input['token_type_ids'].squeeze(1).to(device)
        attention_mask = train_input['attention_mask'].squeeze(1).to(device)
        label = train_label.long().to(device)
        length = len(token_ids)
        output = model(token_ids, segment_ids, attention_mask,length)
        loss = loss_fn(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_acc += accuracy(output, label)
        train_ce += loss.data.cpu().numpy()
        train_precision += precision_score(output,label).cpu().numpy()
        train_recall += recall_score(output,label).cpu().numpy()
        train_f1 += f1(output,label).cpu().numpy()  
        if (batch_id+1) % len(train_dataloader) == 0:
                print("epoch {} train loss {} train acc {} train precision {} train recall {} train f1 {}".format(e+1,train_ce/(batch_id+1), 
                                                                                                                  train_acc / (batch_id+1),
                                                                                                                  train_precision / (batch_id+1),
                                                                                                                  train_recall / (batch_id+1),
                                                                                                                  train_f1/(batch_id+1)))
                train_acc_L.append(train_acc/(batch_id+1))
                train_ce_L.append(train_ce/(batch_id+1))
        
 # Test      
    model.eval()
    for batch_id, (test_input, test_label) in enumerate((test_dataloader)):
        token_ids = test_input['input_ids'].squeeze(1).to(device)
        segment_ids = test_input['token_type_ids'].squeeze(1).to(device)
        attention_mask = test_input['attention_mask'].squeeze(1).to(device)
        label = test_label.long().to(device)
        length = len(token_ids)
        output = model(token_ids,segment_ids,attention_mask,length = len(token_ids))
        loss = loss_fn(output,label)
        test_acc += accuracy(output, label)
        test_ce += loss.data.cpu().numpy()
        test_precision += precision_score(output,label).cpu().numpy()
        test_recall += recall_score(output,label).cpu().numpy()
        test_f1 += f1(output,label).cpu().numpy()
        if (batch_id+1) % len(test_dataloader) == 0:
            print("epoch {} test loss {} test acc {} test precision {} test recall {} test f1 {}".format(e+1,test_ce/(batch_id+1),
                                                                                                             test_acc / (batch_id+1),
                                                                                                             test_precision / (batch_id+1),
                                                                                                             test_recall / (batch_id+1),
                                                                                                             test_f1/(batch_id+1)))
            test_acc_L.append(test_acc/(batch_id+1))
            test_ce_L.append(test_ce/(batch_id+1))
            test_precision_L.append(test_precision/(batch_id+1))
            test_recall_L.append(test_recall/(batch_id+1))
            test_f1_score.append(test_f1/(batch_id+1))
            
end = time.time() - start
print(end)
print('Fianl ACC : {} Final precision {} Final Recall {} Final F1 : {}'.format(np.mean(test_acc_L),
                                                                                   np.mean(test_precision_L),
                                                                                   np.mean(test_recall_L),
                                                                                   np.mean(test_f1_score)))
MRD_AVM = MRD_AVM.append({'Num_Topic' : num_topic,
                 'AUC' : np.mean(test_acc_L),
                 'Precision' : np.mean(test_precision_L),
                 'Recall' : np.mean(test_recall_L),
                 'F1' : np.mean(test_f1_score),
                 'random_seed' : random_seed,
                 'Time' : end},ignore_index = True)
