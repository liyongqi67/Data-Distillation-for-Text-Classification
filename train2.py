import torch
import os
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F

from network import textCNN

from tqdm import tqdm

from evaluation import evaluate_steps
from itertools import chain

import pickle
import torchtext
import csv
import re
import random


vec =torchtext.vocab.GloVe(name="twitter.27B", dim="100",cache="/home/liyongqi/data_dir/dataset distillation/word2vectors/")

class_num=4
batchSize=256
pattern_len = 80

textCNN_param = {
    'vocab_size': 609611,
    'embed_dim': 100,
    'class_num': class_num,
    "kernel_num": 128,
    "kernel_size": [2, 3],
    "dropout": 0.5,
}





os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# cpu
torch.manual_seed(67)
# gpu
torch.cuda.manual_seed_all(67) 




def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r"\"", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()






train_data=[]

with open('/home/liyongqi/data_dir/dataset distillation/AG_NEWS/ag_news_csv/train.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:

        text=clean_str(row[2]).strip().split(" ")

        temp=[]
        for s in text:
            temp.append(s) 
        class_index=int(row[0].strip())-1
        train_data.append([class_index,temp])



test_data=[]
with open('/home/liyongqi/data_dir/dataset distillation/AG_NEWS/ag_news_csv/test.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        text=clean_str(row[2]).strip().split(" ")
        temp=[]
        for s in text:
            temp.append(s) 
        class_index=int(row[0].strip())-1
        test_data.append([class_index,temp])


print('len train_data',len(train_data))
print('len test_data',len(test_data))
random.shuffle(train_data)
random.shuffle(test_data)
print('load data end')

#train_data=train_data[:int(0.001*len(train_data))]
#print('len train_data',len(train_data))
def getTrainBatch(batchSize,num):    
    pattern=np.zeros([batchSize,pattern_len,textCNN_param['embed_dim']],dtype=np.float)
    
    tag=np.zeros([batchSize],dtype=np.int)


    for i in range(num*batchSize,(num+1)*batchSize):
        text=train_data[i][1][:pattern_len]+['0']*(max(pattern_len,len(train_data[i][1]))-len(train_data[i][1]))


        pattern[i%batchSize]=vec.get_vecs_by_tokens(text, lower_case_backup=True)



        tag[i%batchSize]=train_data[i][0]
#        tag[i%batchSize]=train_data[i][0]


    return pattern,tag

def getTestBatch(batchSize,num):   
    pattern=np.zeros([batchSize,pattern_len,textCNN_param['embed_dim']],dtype=np.float)

    tag=np.zeros([batchSize],dtype=np.int)
    for i in range(num*batchSize,(num+1)*batchSize):
        text=test_data[i][1][:pattern_len]+['0']*(max(pattern_len,len(test_data[i][1]))-len(test_data[i][1]))
        pattern[i%batchSize]=vec.get_vecs_by_tokens(text, lower_case_backup=True)

        tag[i%batchSize]=test_data[i][0]


    return pattern,tag

def eva(net):
    net.eval()
    valLoop=len(test_data)/batchSize
    valLoop=int(valLoop)


    print("start eva")
    count=0
    with torch.no_grad():
        for valNum in tqdm((range(valLoop))):

            pattern,tag=getTestBatch(batchSize,valNum)
            pattern=torch.from_numpy(pattern)

            
            pattern=pattern.cuda()



            out=net(pattern)



            score=np.argmax(out.cpu().numpy(),axis=1)

            for i in range(batchSize):

                if score[i]==tag[i]:
                    count+=1

    print('count',count)
    print('len test',(batchSize*valLoop))
    acc=float(count)/(batchSize*valLoop)
    print('acc',acc)
    return acc


    #init net
print('init net...')
net = textCNN(textCNN_param)

net.cuda()
for parameters in net.parameters():
    print(parameters)
# 保存整个网络
torch.save(net, "net.pth")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

class State():
    def __init__(self, models,test_loader_iter):
        self.test_models=models
        self.test_loader_iter=test_loader_iter
        self.device=torch.device("cuda:0")
        self.num_classes=class_num
        self.test_nets_type="111"

class Trainer(object):
    def __init__(self, models):

        self.device = torch.device("cuda:0")

        self.models = models
        self.num_data_steps = 27 # how much data we have
        self.distill_epochs=6
        self.T = self.num_data_steps * self.distill_epochs  # how many sc steps we run
        self.num_classes=class_num
        self.distilled_images_per_class_per_step=1
        self.num_per_step = self.num_classes * self.distilled_images_per_class_per_step

        self.distill_lr=0.01




        self.init_data_optim()

    def init_data_optim(self):
        self.params = []

        optim_lr = 0.01

        # labels
        self.labels = []
        distill_label = torch.arange(self.num_classes, dtype=torch.long, device=self.device) \
                             .repeat(self.distilled_images_per_class_per_step, 1)  # [[0, 1, 2, ...], [0, 1, 2, ...]]
        distill_label = distill_label.t().reshape(-1)  # [0, 0, ..., 1, 1, ...]
        for _ in range(self.num_data_steps):
            self.labels.append(distill_label)
        self.all_labels = torch.cat(self.labels)

        # data
        self.data = []
        for _ in range(self.num_data_steps):
            distill_data = torch.randn(self.num_per_step, pattern_len, textCNN_param['embed_dim'],
                                       device=self.device, requires_grad=True)
            self.data.append(distill_data)
            self.params.append(distill_data)

        # lr

        # undo the softplus + threshold
        raw_init_distill_lr = torch.tensor(self.distill_lr, device=self.device)
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.T, 1)
        self.raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()
        #self.params.append(self.raw_distill_lrs)




        self.optimizer = torch.optim.Adam(self.params, lr=optim_lr)
        # self.optimizer = torch.optim.SGD(self.params, lr=optim_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=2,
                                                   gamma=0.5)
        for p in self.params:
            p.grad = torch.zeros_like(p)

    def get_steps(self):
        data_label_iterable = (x for _ in range(self.distill_epochs) for x in zip(self.data, self.labels))
        lrs = F.softplus(self.raw_distill_lrs).unbind()

        steps = []
        for (data, label), lr in zip(data_label_iterable, lrs):
            steps.append((data, label, lr))

        return steps

    def forward(self, model, rdata, rlabel, steps):
  
        # forward
        model.train()
        w = model.get_param()

        params = [w]
        gws = []

        for step_i, (data, label, lr) in enumerate(steps):
            with torch.enable_grad():

                output = model.forward_with_param(data, w)
                loss = nn.NLLLoss()(output, label)
            gw, = torch.autograd.grad(loss, w, lr.squeeze(), create_graph=True)

            with torch.no_grad():
                new_w = w.sub(gw).requires_grad_()
                params.append(new_w)
                gws.append(gw)
                w = new_w

        # final L
        model.eval()
        output = model.forward_with_param(rdata, params[-1])
        ll = nn.NLLLoss() (output, rlabel)
        return ll, (ll, params, gws)

    def backward(self, model, rdata, rlabel, steps, saved_for_backward):
        l, params, gws = saved_for_backward


        datas = []
        gdatas = []
        lrs = []
        glrs = []

        dw, = torch.autograd.grad(l, (params[-1],))

        # backward
        model.train()
        # Notation:
        #   math:    \grad is \nabla
        #   symbol:  d* means the gradient of final L w.r.t. *
        #            dw is \d L / \dw
        #            dgw is \d L / \d (\grad_w_t L_t )
        # We fold lr as part of the input to the step-wise loss
        #
        #   gw_t     = \grad_w_t L_t       (1)
        #   w_{t+1}  = w_t - gw_t          (2)
        #
        # Invariants at beginning of each iteration:
        #   ws are BEFORE applying gradient descent in this step
        #   Gradients dw is w.r.t. the updated ws AFTER this step
        #      dw = \d L / d w_{t+1}
        for (data, label, lr), w, gw in reversed(list(zip(steps, params, gws))):
            # hvp_in are the tensors we need gradients w.r.t. final L:
            #   lr (if learning)
            #   data
            #   ws (PRE-GD) (needed for next step)
            #
            # source of gradients can be from:
            #   gw, the gradient in this step, whose gradients come from:
            #     the POST-GD updated ws
            hvp_in = [w]
            hvp_in.append(data)
            hvp_in.append(lr)
            dgw = dw.neg()  # gw is already weighted by lr, so simple negation
            hvp_grad = torch.autograd.grad(
                outputs=(gw,),
                inputs=hvp_in,
                grad_outputs=(dgw,)
            )
            # Update for next iteration, i.e., previous step
            with torch.no_grad():
                # Save the computed gdata and glrs
                datas.append(data)
                gdatas.append(hvp_grad[1])
                lrs.append(lr)
                glrs.append(hvp_grad[2])

                # Update for next iteration, i.e., previous step
                # Update dw
                # dw becomes the gradients w.r.t. the updated w for previous step
                dw.add_(hvp_grad[0])

        return datas, gdatas, lrs, glrs



    def accumulate_grad(self, grad_infos):
        bwd_out = []
        bwd_grad = []
        for datas, gdatas, lrs, glrs in grad_infos:
            bwd_out += list(lrs)
            bwd_grad += list(glrs)
            for d, g in zip(datas, gdatas):
                d.grad.add_(g)
        if len(bwd_out) > 0:
            torch.autograd.backward(bwd_out, bwd_grad)



    def __call__(self):
        return self.train()

    def train(self):

        bestPerformance=0
        for epoch in range(5):
            for it in (range(int(len(train_data)/batchSize))):

                pattern,tag=getTrainBatch(batchSize,it)


                pattern=torch.from_numpy(pattern)
                tag=torch.from_numpy(tag)
                
                pattern=pattern.cuda()
                tag=tag.cuda()


                # if it == 0:
                #     self.scheduler.step()

                if it == 0 and epoch == 0:
                    with torch.no_grad():
                        steps = self.get_steps()

                self.optimizer.zero_grad()
                rdata, rlabel = pattern.to(self.device, non_blocking=True), tag.to(self.device, non_blocking=True)


                tmodels = self.models

                t0 = time.time()
                losses = []
                steps = self.get_steps()



                # activate everything needed to run on this process
                grad_infos = []
                for model in tmodels:

                    l, saved = self.forward(model, rdata, rlabel, steps)
                    losses.append(l.detach())
                    grad_infos.append(self.backward(model, rdata, rlabel, steps, saved))
                    del l, saved
                self.accumulate_grad(grad_infos)


                all_reduce_tensors = [p.grad for p in self.params]
                if (it%100==1):
                    losses = torch.stack(losses, 0).sum()
                    all_reduce_tensors.append(losses)
                    loss = losses.item()
                    print('epoch',epoch,'it',it,'loss',loss)


                # opt step
                self.optimizer.step()




                del steps, grad_infos, losses, all_reduce_tensors

            ######每个epoch后测试
 
            network = torch.load( "net.pth") 

            with torch.no_grad():
                steps = self.get_steps()
            for parameters in network.parameters():
                print(parameters)

            for step_i, (data, label, lr) in enumerate(steps):

                data = data.detach()
                label = label.detach()
                lr = lr.detach()

                optimizer = torch.optim.SGD(network.parameters(), lr=lr.item())
                network.train()

                pattern=data.cuda()
                tag=label.cuda()


                optimizer.zero_grad()

                out=network(pattern)

                loss = nn.NLLLoss()(out, tag)
                loss.backward()
                optimizer.step()

                print("step_i:",step_i, 'lr',lr.item(),'loss:',loss.item())
            for parameters in network.parameters():
                print(parameters)
            acc=eva(network)
            if(acc>bestPerformance):
                bestPerformance=acc
                torch.save(steps,"steps")
            print('acc',acc)
            print('bestPerformance',bestPerformance)









Trainer([net]).train()



# *****************************************************************************************
#**************用学习的lr
# steps=torch.load("steps")

# for step_i, (data, label, lr) in enumerate(steps):

#     data = data.detach()
#     label = label.detach()
#     lr = lr.detach()

#     optimizer = torch.optim.SGD(net.parameters(), lr=lr.item())
#     net.train()

#     pattern=data.cuda()
#     tag=label.cuda()


#     optimizer.zero_grad()

#     out=net(pattern)

#     loss = nn.NLLLoss()(out, tag)
#     loss.backward()
#     optimizer.step()

#     print("step_i:",step_i, 'lr',lr.item(),'loss:',loss.item())
#     if((step_i+1)%30==0):
#         acc=eva(net)
#         print('acc',acc)        

# acc=eva(net)
# print('acc',acc)

# for parameters in net.parameters():
#     print(parameters)



#*****************************************************************************************
#用自定义的lr和SGD
# steps=torch.load("steps")

# bestPerformance=0
# for epoch in range(20):
#     for step_i, (data, label, lr) in enumerate(steps):

#         data = data.detach()
#         label = label.detach()
#         lr = lr.detach()

#         optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
#         net.train()

#         pattern=data.cuda()
#         tag=label.cuda()


#         optimizer.zero_grad()

#         out=net(pattern)

#         loss = nn.NLLLoss()(out, tag)
#         loss.backward()
#         optimizer.step()

#     acc=eva(net)
#     if(acc>bestPerformance):
#         bestPerformance=acc
#     print('acc',acc)
#     print('bestPerformance',bestPerformance)


#*****************************************************************************************
#用自定义的lr和Adam
# steps=torch.load("steps")
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# bestPerformance=0
# for epoch in range(100):
#     for step_i, (data, label, lr) in enumerate(steps):

#         data = data.detach()
#         label = label.detach()
#         lr = lr.detach()

#         net.train()

#         pattern=data.cuda()
#         tag=label.cuda()


#         optimizer.zero_grad()

#         out=net(pattern)

#         loss = nn.NLLLoss()(out, tag)
#         loss.backward()
#         optimizer.step()

#     acc=eva(net)
#     if(acc>bestPerformance):
#         bestPerformance=acc
#     print('acc',acc)
#     print('bestPerformance',bestPerformance)



# ****************************************************

#evaluate_steps测试目前看效果等于SGD加学习的lr


# steps=torch.load("steps")
# test_loader_iter=[]
# valLoop=len(test_data)/batchSize
# valLoop=int(valLoop)

# for valNum in tqdm((range(valLoop))):

#     pattern,tag=getTestBatch(batchSize,valNum)
#     pattern=torch.from_numpy(pattern)    
#     pattern=pattern.cuda()
#     pattern=embed(pattern)
#     tag=torch.from_numpy(tag)    
#     tag=tag.cuda()


#     test_loader_iter.append((pattern,tag))
# state=State([net],test_loader_iter)


# evaluate_steps(state, steps, 'Begin of epoch {}'.format(3))







# *****************************************************************************************
#正常训练


# print(len(train_data))

# train_data=train_data[:14024]
# print(len(train_data))


# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


# bestPerformance=0
# print("training...")
# for epoch in range(40):
#     for num in (range(int(len(train_data)/batchSize))):
#         net.train()
#         pattern,tag=getTrainBatch(batchSize,num)

#         pattern=torch.from_numpy(pattern)
#         tag=torch.from_numpy(tag)
        
#         pattern=pattern.cuda()
#         tag=tag.cuda()


#         optimizer.zero_grad()

#         out=net(embed(pattern))

#         loss = nn.NLLLoss()(out, tag)
#         loss.backward()
#         optimizer.step()

#         if num%100==1:
#             print("epoch:",epoch,' num:',num,' loss:',loss.item())

#     acc=eva(net)
#     if(acc>bestPerformance):
#         bestPerformance=acc
#     print('acc',acc)
#     print('bestPerformance',bestPerformance)



# *****************************************************************************************
#构建字典将有label的relation映射到0-1703

# count=0
# class_dict={}

# for num in (range(int(len(train_data)/batchSize))):
#     pattern,tag=getTrainBatch(batchSize,num)
#     tag=list(tag)
#     for s in tag:
#         if s not in class_dict:
#             class_dict[s]=count
#             count+=1
# for num in (range(int(len(test_data)/batchSize))):
#     pattern,tag=getTestBatch(batchSize,num)
#     tag=list(tag)
#     for s in tag:
#         if s not in class_dict:
#             class_dict[s]=count
#             count+=1
# with open('class_dict.pkl', 'wb') as f:
#     pickle.dump(class_dict, f)
# print(count)


