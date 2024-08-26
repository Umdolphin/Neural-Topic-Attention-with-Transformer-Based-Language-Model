# MRD_ANTM

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
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

# Set Hyperparameter 
window_size = 5
min_count = 5
max_word_num = 500
reduced_vocab_size = 2000
num_classes= 3
topic_hidden_size=64
dropout_keep_proba=0.1
hidden_size = 768

learning_rate =  2e-5
num_epochs = 5
max_grad_norm = 1

num_topic = 50
embedding_size = 100
threshold = 0.1

# Make storage
MRD_ANTM = pd.DataFrame(columns = ['Num_Topic','AUC','Precision','Recall','F1','random_seed','Time'])

## Data load

train_df = pd.read_csv('train_MRD_3label.csv')
train_df = train_df[['text','rating']]

test_df = pd.read_csv('test_MRD_3label.csv')
test_df = test_df[['text','rating']]

train_data_label = train_df['rating'].apply(lambda x: int(x))
train_data_label = train_data_label.tolist()

test_data_label = test_df['rating'].apply(lambda x: int(x))
test_data_label = test_data_label.tolist()

## Train word2vec
train_data_text = train_df['text'].apply(lambda x: x.lower())
train_data_lower = train_data_text.str.replace("[^a-zA-Z0-9]"," ")
train_data_join = train_data_lower.apply(lambda x: ' '.join(x.split()))
train_data_word_tok = train_data_join.apply(lambda x: word_tokenize(x))
train_data_final = train_data_word_tok.apply(lambda x: [w for w in x if w not in stopWords]).tolist()

test_data_text = test_df['text'].apply(lambda x: x.lower())
test_data_lower = test_data_text.str.replace("[^a-zA-Z0-9]"," ")
test_data_join = test_data_lower.apply(lambda x: ' '.join(x.split()))
test_data_word_tok = test_data_join.apply(lambda x: word_tokenize(x))
test_data_final = test_data_word_tok.apply(lambda x: [w for w in x if w not in stopWords]).tolist()

w2v_data = train_data_final

# Fit Word2vec  

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = Word2Vec(sentences=w2v_data, vector_size=embedding_size, window=window_size, min_count=min_count, workers=2, sg=0)

w2v = []
for i in range(len(model.wv)):
    weight = model.wv[i]
    word = model.wv.index_to_key[i]
    w2v.append([word,weight])
    
# To build a Word2Vec (w2v) dictionary and use reduced vocabulary for Neural Topic Model input.

def make_vocab(word_weight_vector,w2v_model):
    word_matrix = []
    old_vocab = w2v_model.wv.index_to_key
    
    for i in range(len(word_weight_vector)):
        word_matrix.append(word_weight_vector[i][1])

    wv_matrix_array = np.asarray(word_matrix)
    vocab_dict = dict(zip(old_vocab,range(1,len(old_vocab)+1)))
    
    return wv_matrix_array,vocab_dict,old_vocab

wv_matrix, vocab_dict, old_vocab = make_vocab(w2v,model)


# Remove stopwords and reduce the vocabulary size using a TF-IDF matrix
# Data preprocessing -> Final input format
# Save word file


f = open('train_data_word.txt','w')

for i,word in enumerate(train_data_final):
    if i>=1:
        f.write('\n')
    for j in range(len(word)):
        check_final = len(word)
        if j==0:
            f.write(str(train_data_label[i]))
            f.write('.')
        elif (check_final-1) == j:
                f.write(word[j])
        else:
            f.write(word[j])
            f.write(',')
f.close()

t = open('test_data_word.txt','w')

for i,word in enumerate(test_data_final):
    if i>=1:
        t.write('\n')
    for j in range(len(word)):
        check_final = len(word)
        if j==0:
            t.write(str(test_data_label[i]))
            t.write('.')
        elif (check_final-1) == j:
            t.write(word[j])
            else:
            t.write(word[j])
            t.write(',')
t.close()

def decrease_vocab(path, vocab, target_vocab_size=2000):
    _vocab = [w for w in vocab if w not in stopWords]
    docs = []
    myFile = open(path,'rU')
    for i,Row in enumerate(myFile):
        text = Row.split('.')[1]
        text = text[:-1]
        docs.append(text)
    myFile.close()
    vectorizer = TfidfVectorizer(vocabulary = _vocab)
    X = vectorizer.fit_transform(docs)
    word_importance = np.sum(X, axis = 0) # shape: [1, vocab_size], a numpy matrix!
    sorted_vocab_idx = np.squeeze(np.asarray(np.argsort(word_importance), dtype=np.int32)) # shape: [vocab_size, ], a numpy array
    vocab_idx_wanted = np.flip(sorted_vocab_idx)[:target_vocab_size] # decending order, int
    new_vocab = [_vocab[i] for i in vocab_idx_wanted]
    new_vocab_dict = dict(zip(new_vocab, range(target_vocab_size)))
    with open("topic_model_vocab.txt", 'w') as w_f:
            w_f.write('\n'.join(new_vocab))
    return new_vocab, new_vocab_dict

decrease_vocab('train_data_word.txt',old_vocab)

vocab_size = 2000
num_classes = 3

vocab = []
with open("topic_model_vocab.txt") as read_f:
    for line in read_f:
        vocab.append(line.strip())
        

# Convert the data into an input format suitable for the topic model

MAX_NUM_WORD = 500
EMBEDDING_SIZE = embedding_size

data_train_path = ('train_data_word.txt')
data_test_path = ('test_data_word.txt')

def read_topical_atten_data(path, vocab_dict, reduced_vocab):
    myFile = open(path, "rU")
    labels = []
    docIDs = []
    docs = []
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    
    for i, aRow in enumerate(myFile):
        line = aRow.strip().split('.')
        ids = []
        for w in line[1].split(','):
            if w in vocab_dict:
                ids.append(vocab_dict[w])
        labels.append(int(line[0]))
        docIDs.append(ids)
        docs.append(line[1])
    
    myFile.close()
    
    num_docs = len(labels)
    print(num_docs, "docs in total")
    
    y = np.zeros((num_docs, num_classes), dtype=np.int32)
    x = np.zeros((num_docs, MAX_NUM_WORD), dtype=np.int32)
    
    for i in range(num_docs):
        y[i][int(labels[i])-1] = 1
        if len(docIDs[i]) > MAX_NUM_WORD:
            x[i, :] = docIDs[i][:MAX_NUM_WORD]
        else:
            x[i, :len(docIDs[i])] = docIDs[i]
    
    counts = count_vect.fit_transform(docs).toarray()
    
    return x, y, counts, num_docs


train_x_rnn, train_y, train_x_bow, num_train_docs = read_topical_atten_data(data_train_path, vocab_dict, vocab)
test_x_rnn, test_y, test_x_bow, num_test_docs = read_topical_atten_data(data_test_path, vocab_dict, vocab)
    
class CustomDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        self.sentences = [bert_tokenizer(text, padding='max_length', return_tensors="pt", max_length=max_len, truncation=True) for text in dataset['text']]
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
        return len(self.labels)

pretrained_embed = wv_matrix

    

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    empty_torch = torch.zeros(fan_in,fan_out,dtype = torch.float32)
    return torch.nn.init.uniform_(empty_torch,
                             a=low, b=high)

class VariationalTopicModel(nn.Module):

    def __init__(self, vocab_size, latent_dim, num_topic, embedding_size, dropout_keep_proba=0.1):
        super(VariationalTopicModel, self).__init__()
        self.num_topic = num_topic
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.embedding_size = embedding_size
        self.dropout_keep_proba = dropout_keep_proba 

        # Encoder
        self.encoder_fc1 = nn.Linear(self.vocab_size, self.latent_dim)
        self.encoder_fc2 = nn.Linear(self.latent_dim, self.latent_dim)
        self.batch_norm = nn.BatchNorm1d(self.latent_dim)
        self.dropout1 = nn.Dropout(self.dropout_keep_proba)
        self.mu_fc = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_fc = nn.Linear(self.latent_dim, self.latent_dim)
        self.dropout2 = nn.Dropout(self.dropout_keep_proba)
        self.topic_fc = nn.Linear(self.latent_dim, self.num_topic)
        
        self.W_fc = nn.Linear(self.embedding_size, self.vocab_size)
        nn.init.xavier_uniform_(self.W_fc.weight)
        self.dropout2 = nn.Dropout(self.dropout_keep_proba)
        self.topic_embed_fc = nn.Linear(self.embedding_size, self.num_topic)
        nn.init.xavier_uniform_(self.topic_embed_fc.weight)     

    def forward(self, x_bow):     

        # Encoder
        hidden_encoder_1 = F.relu(self.encoder_fc1(x_bow))
        hidden_encoder_2 = F.relu(self.encoder_fc2(hidden_encoder_1))
        batch_norm_encoder = self.batch_norm(hidden_encoder_2)
        dropout_encoder = self.dropout1(batch_norm_encoder)
        mu_encoder = self.mu_fc(dropout_encoder)
        logvar_encoder = self.logvar_fc(dropout_encoder)
        epsilon = torch.randn(logvar_encoder.size(), dtype=torch.float32).to(device)
        std_dev = torch.sqrt(torch.exp(logvar_encoder))
        h = mu_encoder + std_dev * epsilon

        topic = F.softmax(self.topic_fc(h), dim=1)
        W = self.W_fc.weight
        W_drop = self.dropout2(W)
        topic_embed = self.topic_embed_fc.weight
        beta = F.softmax(torch.matmul(topic_embed, torch.transpose(W_drop, 0, 1)), dim=1)
        p_x = torch.matmul(topic, beta)

        return topic, topic_embed, mu_encoder, logvar_encoder, p_x



class Topic_Paper_AlbertClassifier(nn.Module):

    def __init__(self, Albert, reduced_vocab_size=2000, num_topic=50, num_classes=3, pretrained_embed=None,
                 dropout_keep_proba=0.1, topic_hidden_size=64, embedding_size=100, hidden_size=768,
                 threshold=0.1, dr_rate=None, params=None):

        super(Topic_Paper_AlbertClassifier, self).__init__()
        self.threshold = threshold
        self.vtm = VariationalTopicModel(reduced_vocab_size, topic_hidden_size, num_topic, embedding_size, dropout_keep_proba)
        self.Albert = Albert
        self.dr_rate = dr_rate
        self.h1_fc = nn.Linear(hidden_size, embedding_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
        # nn.init.xavier_uniform_(self.h1_fc.weight)
        # nn.init.xavier_uniform_(self.classifier.weight)
        
    def forward(self, token_ids, segment_ids, attention_mask, bow_x):
        
        last_hidden_states = self.Albert(input_ids=token_ids, token_type_ids=segment_ids, attention_mask=attention_mask)[0]
        
        topic, topic_embed, mu_encoder, logvar_encoder, p_x = self.vtm.forward(bow_x)
        w = topic - self.threshold
        topic_embed_unstack = torch.unbind(topic_embed)
        topic_atten_weights = []
        outputs = last_hidden_states
        h1 = torch.tanh(self.h1_fc(outputs)).to(device)

        for i in range(num_topic):
            query = topic_embed_unstack[i]
            score = torch.sum(torch.multiply(h1, query), dim=-1, keepdim=True)
            attention_weights = F.softmax(score, dim=1)
            topic_atten_weights.append(attention_weights)
        
        topic_atten = torch.matmul(torch.concat(topic_atten_weights, -1), w.unsqueeze(-1))
        atten_out = torch.sum(torch.multiply(topic_atten, outputs), dim=1)
        
        if self.dr_rate:
            out = self.dropout(atten_out)
        
        return self.classifier(out), topic, topic_embed, mu_encoder, logvar_encoder, p_x

    
device = torch.device("cuda")

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
Albert = AlbertModel.from_pretrained('albert-base-v2')

max_len = 256
    
import re
train_df['text'] = train_df['text'].apply(lambda x: re.sub('[^a-zA-Z0-9]',' ',x).strip())
train_df['text'] = train_df['text'].apply(lambda x: ' '.join(x.split()))
test_df['text'] = test_df['text'].apply(lambda x: re.sub('[^a-zA-Z0-9]',' ',x).strip())
test_df['text'] = test_df['text'].apply(lambda x: ' '.join(x.split()))

batch_size = 8
train_dataset = CustomDataset(train_df, 0,1, tokenizer, max_len, True, False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_dataset = CustomDataset(test_df, 0,1, tokenizer, max_len, True, False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    

final_model = Topic_Paper_AlbertClassifier(Albert,reduced_vocab_size = reduced_vocab_size, num_topic = num_topic,
                                           num_classes = num_classes, pretrained_embed = pretrained_embed, dropout_keep_proba = dropout_keep_proba,
                                           topic_hidden_size = topic_hidden_size, embedding_size = embedding_size,hidden_size = hidden_size,
                                           threshold = threshold,dr_rate = 0.1).to(device)
    
    
# Adamw

t_total = len(train_dataloader) * num_epochs

loss_fn = nn.CrossEntropyLoss()

warmup_ratio = 0.1
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in final_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in final_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
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

for e in range(num_epochs):
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
    
    final_model.train()
    train_batch_total = len(train_x_bow) // batch_size
    print("Current epochs: {}".format(e))

    for i, (train_input, train_label) in zip(range(train_batch_total), train_dataloader):
        optimizer.zero_grad()
        train_x_batch_rnn = torch.tensor(train_x_rnn[i * batch_size: (i + 1) * batch_size]).to(device)
        train_x_batch_bow = torch.tensor(train_x_bow[i * batch_size: (i + 1) * batch_size], dtype=torch.float32).to(device)
        token_ids = train_input['input_ids'].squeeze(1).to(device)
        segment_ids = train_input['token_type_ids'].squeeze(1).to(device)
        attention_mask = train_input['attention_mask'].squeeze(1).to(device)
        label = train_label.long().to(device)
        out, topic, topic_embed, mu_encoder, logvar_encoder, p_x = final_model(token_ids, segment_ids, attention_mask, train_x_batch_bow)

        kl_divergence = -0.5 * torch.sum(1.0 + logvar_encoder - torch.square(mu_encoder) - torch.exp(logvar_encoder), 1)
        likelihood = -torch.sum(torch.multiply(torch.log(p_x + 1e-10), train_x_batch_bow), 1)

        var_loss_inference = likelihood + kl_divergence
        kl_divergence_mean = torch.mean(kl_divergence)
        likelihood_mean = torch.mean(likelihood)
        var_loss = torch.mean(var_loss_inference)

        num_words = torch.sum(train_x_batch_bow, axis=-1)
        perp = torch.exp(torch.mean(torch.true_divide(likelihood, num_words)))

        loss_BERT = loss_fn(out, label)
        loss = var_loss + loss_BERT
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_acc += accuracy(out, label)
        train_ce += loss_BERT.data.cpu().numpy()
        train_precision += precision_score(out, label).cpu().numpy()
        train_recall += recall_score(out, label).cpu().numpy()
        train_f1 += f1(out, label).cpu().numpy()

        if (i + 1) % train_batch_total == 0:
            print("epoch {} train loss {} train acc {} train precision {} train recall {} train f1 {}".format(
                e + 1, train_ce / (i + 1),
                train_acc / (i + 1),
                train_precision / (i + 1),
                train_recall / (i + 1),
                train_f1 / (i + 1))
            )

# Test            
            
    final_model.eval()
    test_batch_total = len(test_x_bow) // batch_size
    for i, (test_input, test_label) in zip(range(test_batch_total), test_dataloader):
        test_x_batch_rnn = torch.tensor(test_x_rnn[i * batch_size: (i + 1) * batch_size]).to(device)
        test_x_batch_bow = torch.tensor(test_x_bow[i * batch_size: (i + 1) * batch_size], dtype=torch.float32).to(device)
        token_ids = test_input['input_ids'].squeeze(1).to(device)
        segment_ids = test_input['token_type_ids'].squeeze(1).to(device)
        attention_mask = test_input['attention_mask'].squeeze(1).to(device)
        label = test_label.long().to(device)
        out, topic, topic_embed, mu_encoder, logvar_encoder, p_x = final_model(token_ids, segment_ids, attention_mask, test_x_batch_bow)
        var_loss_mean = torch.mean(var_loss)
        loss_BERT = loss_fn(out, label)
        loss = var_loss_mean + loss_BERT
        test_acc += accuracy(out, label)
        test_ce += loss_BERT.data.cpu().numpy()
        test_precision += precision_score(out, label).cpu().numpy()
        test_recall += recall_score(out, label).cpu().numpy()
        test_f1 += f1(out, label).cpu().numpy()

        if (i + 1) % test_batch_total == 0:
            print("epoch {} test loss {} test acc {} test precision {} test recall {} test f1 {}".format(
                e + 1, test_ce / (i + 1),
                test_acc / (i + 1),
                test_precision / (i + 1),
                test_recall / (i + 1),
                test_f1 / (i + 1))
            )
            test_acc_L.append(test_acc / (i + 1))
            test_ce_L.append(test_ce / (i + 1))
            test_precision_L.append(test_precision / (i + 1))
            test_recall_L.append(test_recall / (i + 1))
            test_f1_score.append(test_f1 / (i + 1))
            print('\n')
                
            
end = time.time() - start
print(end)
print('Fianl ACC : {} Final precision {} Final Recall {} Final F1 : {}'.format(np.mean(test_acc_L),
                                                                                   np.mean(test_precision_L),
                                                                                   np.mean(test_recall_L),
                                                                                   np.mean(test_f1_score)))
    
MRD_ANTM = MRD_ANTM.append({'Num_Topic' : num_topic,
                 'AUC' : np.mean(test_acc_L),
                 'Precision' : np.mean(test_precision_L),
                 'Recall' : np.mean(test_recall_L),
                 'F1' : np.mean(test_f1_score),
                 'random_seed' : random_seed,
                 'Time' : end},ignore_index = True)
