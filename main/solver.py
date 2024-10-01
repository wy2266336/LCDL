import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from parse import *
import random
from bert_optimizer import BertAdam
from sklearn import metrics
import numpy as np
import torch
import warnings
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import datetime

class Solver():
    def __init__(self, args):
        warnings.filterwarnings('ignore')
        self.args = args
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        #self.device_ids = [0,1]
        self.model_dir = make_save_dir(args.model_dir)
        if not os.path.exists(os.path.join(self.model_dir,'code')):
            os.makedirs(os.path.join(self.model_dir,'code'))
        
        self.data_utils = data_utils(args)
        self.model = self._make_model(self.data_utils.vocab_size)

        self.test_vecs = None
        self.test_masked_lm_input = []



    def _make_model(self, vocab_size, N=8, N_GCN=2, label_N=4, 
            d_model=512, d_ff=2048, h=2, nheads=4, hidden_dim=200, dropout=0.1):#, label_embedding_dim=300, threshold=0, alpha=0.2):
            """
            N:表示transformer的层数
            N_GCN:表示GCN的层数
            label_N:表示标签个数
            d_model:词向量维度
            h:表示多头attention的头数
            """
            "Helper: Construct a model from hyperparameters."
            c = copy.deepcopy
            attn = MultiHeadedAttention(h, d_model)
            group_attn = GroupAttention(d_model)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout)
            word_embed = nn.Sequential(Embeddings(d_model, vocab_size), c(position))
            model = Encoder(EncoderLayer(d_model, c(attn), c(ff), group_attn, dropout), 
                    N, N_GCN, h, nheads, d_model, hidden_dim, label_N, dropout, c(word_embed))
            
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            #return nn.DataParallel(model,device_ids=self.device_ids).cuda()
            return model.cuda()


    def train(self):
        time0=time.time()
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])
        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                #print(name)
                ttt = 1
                for  s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total_param_num:',tt)

        train_data_yielder = self.data_utils.data_yielder()
        print('train batch num',len(train_data_yielder))
        dev_data_yielder = self.data_utils.data_yielder(mtype='dev',is_mask=False)
        print('dev batch num',len(dev_data_yielder))
        optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4, amsgrad=True)
        #optim = BertAdam(self.model.parameters(), lr=self.args.lr)
        #optim = torch.optim.Adadelta(self.model.parameters(),lr=self.args.lr)
        #optim = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=100)
        #optim = nn.DataParallel(optim, device_ids=self.device_ids).cuda()

        
        total_loss = []
        predict_y = []
        golden_y = []
        start = time.time()
        total_step_time = 0.
        total_masked = 0.
        total_token = 0.
        total_batch = 0.
        best_scores = 0.
        best_value = {}
        temp_scores = 0.
        temp_value = {}
        best_start = 0
        is_down = False


        for step in range(self.args.num_step):
            self.model.train()
            step_start = time.time()
            for batch in tqdm(train_data_yielder):
                out = self.model.forward(batch['input'].long(), batch['input_mask'])
            
                loss,prec,y = self.model.masked_lm_loss(out, batch['y'].long())
                predict_y.extend(prec.tolist())
                golden_y.extend(y.tolist())
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                #optim.module.step()
                optim.step()

                total_loss.append(loss.detach().cpu().numpy())

                total_step_time += time.time() - step_start
            
                if total_batch % 100 == 0:
                    #训练集的评价结果
                    train_acc, train_p, train_r, train_f = estimate(predict_y,golden_y)
                    #train_roc = metrics.roc_auc_score(golden_y,predict_y,average='micro')

                    #验证集的评价结果
                    dev_acc, dev_p, dev_r, dev_f, dev_loss, dev_pred, dev_gold = self.evaluate(dev_data_yielder)
                    temp_scores = dev_f
                    temp_value = {'P': dev_p, 'R': dev_r, 'F1': dev_f, 'dev_pred': dev_pred, 'dev_gold': dev_gold}
                    elapsed = time.time() - start
                    print("Epoch Step: %d Train Acc: %.3f Train P: %.3f Train R: %.3f Train F: %.3f \
                             Train Loss: %f Total Time: %f Step Time: %f" % (step, train_acc, train_p, train_r, train_f, \
                             np.mean(total_loss), elapsed, total_step_time))
                    print("Epoch Step: %d Dev Acc: %.3f Dev P: %.3f Dev R: %.3f Dev F: %.3f Dev Loss: %.3f" %
                            (step, dev_acc, dev_p, dev_r, dev_f, np.mean(dev_loss)))
                    self.model.train()
                    print()
                    start = time.time()
                    total_loss = []
                    predict_y = []
                    golden_y = []
                    total_step_time = 0.
                total_batch += 1

            
                if best_scores < temp_scores:
                    best_scores = temp_scores
                    print('saving!!!!')
                
                    model_name = 'model.pth'
                    state = {'step': step, 'state_dict': self.model.state_dict()}
                    torch.save(state, os.path.join(self.model_dir, model_name))
                    print("The best F1 at prensent is : ", best_scores)
                    best_value = temp_value
                    best_start = step
                else:
                    keep_best = step-best_start
                    if keep_best > self.args.early_stop:
                        is_down = True
            if is_down:
                time1=time.time()
                total_time = time1-time0
                total_time = int(round(total_time))
                total_time = str(datetime.timedelta(seconds=total_time))
                print("Training end!")
                #save_result(best_value['dev_pred'],best_value['dev_gold'],self.args.task_name+'_dev')
                print("The best values in Dev dataset are: Precision %f, Recall %f, F1-scores %f" %(best_value['P'], best_value['R'], best_value['F1']))
                print("Total time is ", total_time)
                break

        """
            if step % 1000 == 0:
                print('saving!!!!')
                
                model_name = 'model.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}
                torch.save(state, os.path.join(self.model_dir, model_name))
        """


    def evaluate(self,yielder):
        self.model.eval()
        total_loss = []
        predict_y = []
        golden_y = []
        with torch.no_grad():
            for batch in yielder:
                out = self.model.forward(batch['input'].long(), batch['input_mask'])
                loss, prec, y = self.model.masked_lm_loss(out,batch['y'].long())
                predict_y.extend(prec.tolist())
                golden_y.extend(y.tolist())
                total_loss.append(loss.detach().cpu().numpy())
            dev_acc, dev_p, dev_r, dev_f = estimate(predict_y,golden_y)
            #dev_roc = metrics.roc_auc_score(golden_y,predict_y,average='micro')
        return dev_acc, dev_p, dev_r, dev_f, total_loss, predict_y, golden_y


    def test(self, threshold=0.8):
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.eval()
        batch_size = self.args.batch_size
        test_data_yielder = self.data_utils.data_yielder('test',False)
        total_loss = []
        predict_y = []
        golden_y = []

        result_dir = os.path.join(self.model_dir, 'result/')
        make_save_dir(result_dir)
        self.f_b = open(os.path.join(result_dir,'brackets.json'),'w')
        self.f_t = open(os.path.join(result_dir,'tree.txt'),'w')

        #test_txts = self.data_utils.test_txts
        with torch.no_grad():
            for i, batch in enumerate(test_data_yielder):
                out = self.model.forward(batch['input'].long(), batch['input_mask'])
                loss, prec, y = self.model.masked_lm_loss(out,batch['y'].long())
                predict_y.extend(prec.tolist())
                golden_y.extend(y.tolist())
                total_loss.append(loss.detach().cpu().numpy())

                #txts = test_txts[i*batch_size:(i+1)*batch_size]
                #self.write_parse_tree(txts, break_probs, threshold)

            test_acc, test_p, test_r, test_f = estimate(predict_y,golden_y)
            #test_roc = metrics.roc_auc_score(golden_y,predict_y,average='micro')

            print('Test over, the parse tree has been written in file!')
            print("test Acc: %.3f test P: %.3f test R: %.3f test F: %.3f test Loss: %.3f" %
                            (test_acc, test_p, test_r, test_f, np.mean(total_loss)))
            #save_result(predict_y,golden_y,self.args.task_name)

    def test_sent(self):
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.eval()
        input_id, input_mask, label_emb, mask, text, r_leng = self.data_utils.sentence2id(self.args.text)
        out, _, weight = self.model.forward(input_id.long(), input_mask, label_emb)
        pred = self.model.predict(out)
        pred_label = []
        for i, pred_value in enumerate(pred[0]):
            if pred_value == 1:
                pred_label.append(self.data_utils.id2label[i])

        print(pred_label)

        weight = weight.squeeze(0)
        adjs = [adj.squeeze(0) for adj in torch.split(weight,1,dim=0)]
        real_adjs = [F.softmax(torch.masked_select(adj,mask).view([r_leng,r_leng]),-1) for adj in adjs]
        make_heatmap(text, real_adjs)
        print('saving the heatmap!')
        with open('weight_matrix.txt', 'w') as f:
            f.write(str(real_adjs))
        f.close()
        print('saving the matrix')

def make_heatmap(text, adjs):
    matplotlib.use('Agg')  
    sns.set()
    plt.figure(figsize=(20,20),dpi=120)
    plt.figure(1)
    for i, adj in enumerate(adjs):
        ax = plt.subplot(221+i)
        sns.heatmap(np.asarray(adj.tolist()), ax=ax, cbar=True, yticklabels=text.split(), xticklabels=text.split())
    plt.savefig('result.jpg')
    plt.show()

'''
def make_roc(predict_y,golden_y):
    fpr, tpr, therod = metrics.roc_curve(golden_y,predict_y,pos_label=1)
    print('----------------------')
    print('fpr \t tpr \t thersholds')
    for i, value in enumerate(therod):
        print('%f %f %f' % (fpr[i],tpr[i],value))
    print('----------------------')
    
    roc_auc = metrics.auc(fpr,tpr)
    
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc.jpg')
    plt.show()

def save_result(predict_y, golden_y,task_name):
    result = {'gold':golden_y, 'pred':predict_y}
    filename = task_name+'.json'
    with open(filename,'w') as f:
        json.dump(result, f)
    print('writting the results')
'''