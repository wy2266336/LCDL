from __future__ import print_function
import numpy as np
import random
import json
import os
import re
import sys
import torch
from tqdm import tqdm
from sklearn import metrics
import operator
import torch.autograd as autograd
from nltk.corpus import stopwords
import transformers
from transformers import BertTokenizer
import time

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


def write_json(filename,data):
    with open(filename, 'w') as fp:
        json.dump(data, fp)


def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir



def cc(arr):
    return torch.from_numpy(np.array(arr)).cuda()


def one_hot(indices, depth):
    shape = list(indices.size())+[depth]
    indices_dim = len(indices.size())
    a = torch.zeros(shape,dtype=torch.float).cuda()
    return a.scatter_(indices_dim,indices.unsqueeze(indices_dim),1)


def get_test(test_file):
    txts = []
    labelss = []
    words = []
    labels = []
    max_len = 0
    for line in open(test_file):
        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
            if words:
                txts.append(words)
                labelss.append(labels)
                words = []
                labels = []
        else:
            word, label = line.strip().split('\t')
            word = word.lower()
            word = re.sub('[0-9]+', 'N', word)
            words.append(word)
            labels.append(label)
        
        if len(words) > max_len:
            max_len = len(words)


    print('test number:',len(txts))
    print('test max_len:',max_len)
    return txts, labelss
'''
def generate_A(num_classes, threshold, adj_file):
    adj_result = read_json(adj_file)
    adj_num_dic = adj_result['nums']
    adj_coMatrix = adj_result['adj']
    nums = np.asarray([v for (_,v) in adj_num_dic.items()])
    adj = np.asarray(adj_coMatrix)
    _adj = adj / nums
    _adj[_adj < threshold] = 0
    _adj[_adj >= threshold] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj +np.identity(num_classes, np.int)
    return _adj

def generate_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A,D).t(),D)
    return adj
'''
class data_utils():
    def __init__(self, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size

        self.dict_path = os.path.join(args.model_dir,'dictionary.json')
        self.train_path = args.train_path
        self.dev_path = args.valid_path
        self.test_path = args.test_path
        #self.label_embed_path = args.embed_file
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.eos_id = 0
        self.unk_id = 1
        self.mask_id = 2
        self.cls_id = 3

        self.get_labels()
        if args.train or not os.path.exists(self.dict_path):
            self.process_training_data()
            self.process_dev_data()

        elif args.test:
            self.new_vocab = read_json(self.dict_path)
            self.process_test_data()
        elif args.test_sen:
            self.new_vocab = read_json(self.dict_path)

        print('vocab_size:',len(self.new_vocab))
        
        self.vocab_size = len(self.new_vocab)
        self.index2word = self.vocab_size*[[]]
        for w in self.new_vocab:
            self.index2word[self.new_vocab[w]] = w

    def get_labels(self):
        '''
        labels = ['O', 'B-Chemical', 'I-Chemical', 'B-Disease', 'I-Disease', 'Pad']
        '''
        labels = ['O', 'B-GENE', 'I-GENE', 'Pad']
        '''
        labels = ['O', 'B-Species', 'I-Species', 'Pad']
        '''
        self.label2id = {l:id for id, l in enumerate(labels)}
        self.id2label = {id:l for id, l in enumerate(labels)}
        self.label_len = len(labels)
        self.pad = len(labels)-1

    def process_training_data(self):
        self.training_data = []
        self.training_label = []

        self.new_vocab = dict()
        self.new_vocab['[PAD]'] = 0
        self.new_vocab['[UNK]'] = 1
        self.new_vocab['[MASK]'] = 2
        self.new_vocab['[CLS]'] = 3
        
        #dd = []
        word_count = {}
        words = []
        labels = []
        w_list = []
        l_list = []
        for line in open(self.train_path):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if w_list:
                    w_list = ['[CLS]'] + w_list
                    l_list = ['Pad'] + l_list
                    words.append(w_list)
                    labels.append(l_list)
                    w_list = []
                    l_list = []
            else:
                word, label = line.strip().split('\t')
                sub_words = self.tokenizer.tokenize(word)
                w = sub_words[0]
                word_count[w] = word_count.get(w,0) + 1
                w_list.append(w)
                l_list.append(label)

        for ll in labels:
            l = [self.label2id[ls] for ls in ll]
            self.training_label.append(l)


        for w in word_count:
            if word_count[w] > 1:
                self.new_vocab[w] = len(self.new_vocab)

        for d in words:
            word_list = []
            for w in d:
                if w in self.new_vocab:
                    word_list.append(self.new_vocab[w])
                else:
                    word_list.append(self.unk_id)
            self.training_data.append(word_list)

        write_json(self.dict_path, self.new_vocab)


    def process_dev_data(self):
        self.dev_data = []
        self.dev_label = []

        txts, labels = get_test(self.dev_path)
        #self.dev_label = [int(label) for label in labels]
        for text in txts:
            w_list = []
            for word in text:
                sub_words = self.tokenizer.tokenize(word)
                w = sub_words[0]
                if w in self.new_vocab:
                    w_list.append(self.new_vocab[w])
                else:
                    w_list.append(self.unk_id)
            w_list = [self.new_vocab['[CLS]']] + w_list
            assert len(text)+1 == len(w_list)
            self.dev_data.append(w_list)

        for label in labels:
            label = ['Pad'] + label
            l_list = [self.label2id[l] for l in label]
            self.dev_label.append(l_list)


    def process_test_data(self):
        self.test_data = []
        self.test_label = []

        txts, labels = get_test(self.test_path)
        #self.test_label = [int(label) for label in labels]
        for text in txts:
            w_list = []
            for word in text:
                sub_words = self.tokenizer.tokenize(word)
                w = sub_words[0]
                if w in self.new_vocab:
                    w_list.append(self.new_vocab[w])
                else:
                    w_list.append(self.unk_id)
            w_list = [self.new_vocab['[CLS]']] + w_list
            assert len(text)+1 == len(w_list)
            self.test_data.append(w_list)

        for label in labels:
            label = ['Pad'] + label
            l_list = [self.label2id[l] for l in label]
            self.test_label.append(l_list)



    def make_masked_data(self, indexed_tokens, indexed_labels, seq_length=128, is_mask=True):
        length = len(indexed_tokens)
        masked_vec = np.zeros([seq_length], dtype=np.int32) + self.eos_id
        #origin_vec = np.zeros([seq_length], dtype=np.int32) + self.eos_id
        #target_vec = np.zeros([seq_length], dtype=np.int32) -1
        gold_label = np.zeros([seq_length], dtype=np.int32) + self.pad
        #real_length = np.array([length], dtype=np.int32)

        unknown = 0.
        masked_num = 0.

        length = len(indexed_tokens)
        for i,word in enumerate(indexed_tokens):
            if i >= seq_length:
                break
            #origin_vec[i] = word
            masked_vec[i] = word
                
            #mask words
            if is_mask:
                if random.randint(0,6) == 0:
                    #target_vec[i] = word
                    masked_num += 1

                    rand_num = random.randint(0,9)
                    if rand_num == 0:
                        #keep the word unchange
                        pass
                    elif rand_num == 1:
                        #sample word
                        masked_vec[i] = random.randint(4, self.vocab_size-1)
                    else:
                        masked_vec[i] = self.mask_id

        for i, label in enumerate(indexed_labels):
            if i >= seq_length:
                break
            gold_label[i] = label

        if is_mask:
            if length > 70 or masked_num == 0:
                masked_vec = None
                
        else:
            if length > 70:
                masked_vec = None
                

        return masked_vec, gold_label, length

    def data_yielder(self, mtype='train', is_mask=True):
        #type: 三种模式，是字符串类型，默认是'train',另外两种是'dev'和‘test’
        #is_mask: 是否进行随机掩码，如果mtype是train时，is_mask是设置成True;如果mtype是dev或者test时,is_mask设置成False
        batches = []
        batch = {'input':[],'input_mask':[],'y':[]}
        datas = []
        labels = []
        #label_embedding = self.get_label_embedding()
        #label_embedding = torch.autograd.Variable(torch.from_numpy(label_embedding)).float().detach()
        if mtype == 'train':
            datas = self.training_data
            labels = self.training_label
        elif mtype == 'dev':
            datas = self.dev_data
            labels = self.dev_label
        else:
            datas = self.test_data
            labels = self.test_label

        max_len = 0
        start_time = time.time()
        #print("\nstart epo %d!!!!!!!!!!!!!!!!\n" % (epo))
        for i, (line, label) in enumerate(zip(datas,labels)):
            input_vec, gold_label, length = self.make_masked_data(line, label, self.seq_length, is_mask)

            if input_vec is not None:
                if length > max_len:
                    max_len = length
                batch['input'].append(input_vec)
                batch['input_mask'].append(np.expand_dims(input_vec != self.eos_id, -2).astype(np.int32))
                #batch['real_length'].append(length)
                batch['y'].append(gold_label)

                if len(batch['input']) == self.batch_size or i == len(datas)-1:
                    batch = {k: cc(v) for k, v in batch.items()}
                    batches.append(batch)
                    max_len = 0
                    batch = {'input':[],'input_mask':[],'y':[]}
        end_time = time.time()
        print('\nfinish time %f!!!!!!!!!!!!!!!\n' % (end_time-start_time))
        return batches

    def id2sent(self,indices, test=False):
        sent = []
        word_dict={}
        for w in indices:
            if w != self.eos_id:
                sent.append(self.index2word[w]) 

        return ' '.join(sent)


    def subsequent_mask(self, vec):
        attn_shape = (vec.shape[-1], vec.shape[-1])
        return (np.triu(np.ones((attn_shape)), k=1).astype('uint8') == 0).astype(np.float)

def estimate(pre_list, gold_list):
    '''
    TP,FP,TN,FN = 0,0,0,0
    for p,g in zip(pre_list,gold_list):
        if p == 1 and g == 1:
            TP += 1
        elif p == 1 and g == 0:
            FP += 1
        elif p == 0 and g == 1:
            FN += 1
        else:
            TN += 1
    Acc = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP/(TP+FP+1e-6)
    Recall = TP/(TP+FN)
    F1 = 2*Precision*Recall/(Precision+Recall+1e-6)
    '''
    Acc, Precision, Recall, F1 = 0,0,0,0
    average = 'macro' #average有[None, 'binary' (default), 'micro', 'macro', 'samples','weighted']
    Acc = metrics.accuracy_score(gold_list, pre_list)
    Precision = metrics.precision_score(gold_list,pre_list,average=average)
    Recall = metrics.recall_score(gold_list,pre_list,average=average)
    F1 = 2*Precision*Recall/(Precision+Recall+1e-6)
    return Acc, Precision, Recall, F1