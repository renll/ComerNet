import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import models
import dataloader
import utils 
import dict

import os
import argparse
import time
import math
import json
import collections

#config
parser = argparse.ArgumentParser(description='predict.py')
parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='data/log/norml_mwNestedNOND128NOsaA1F/checkpoint.pt', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', action='store_true',
                    help="load pretrain embedding")
parser.add_argument('-limit', type=int, default=0,
                    help="data limit")
parser.add_argument('-log', default='predict', type=str,
                    help="log directory")
parser.add_argument('-unk', action='store_true',
                    help="replace unk")
parser.add_argument('-memory', action='store_true',
                    help="memory efficiency")
parser.add_argument('-beam_size', type=int, default=1,
                    help="beam search size")

opt = parser.parse_args([])
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
use_cuda = True
if use_cuda:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(opt.seed)
#checkpoint
if opt.restore:
    print('loading checkpoint...\n')    
    checkpoints = torch.load(opt.restore)
    
#data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)
print('loading time cost: %.3f' % (time.time()-start_time))

testset = datas['test']
src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['src']
testloader = dataloader.get_loader(testset, batch_size=1, shuffle=False, num_workers=2)

from pytorch_pretrained_bert import BertModel
from convert_mw import bert,tokenizer,bert_type

pretrain_embed={}
pretrain_embed['slot'] = torch.load('emb_tgt_mw.pt')
    

# model
print('building model...\n')
bmodel = BertModel.from_pretrained(bert_type)
bmodel.eval()
if use_cuda:
    bmodel.to('cuda')
model = getattr(models, opt.model)(config, src_vocab, tgt_vocab, use_cuda,bmodel,
                       pretrain=pretrain_embed, score_fn=opt.score) 

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()

param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]


model.eval()

preds = []
labels = []
joint_preds=[]
joint_labels=[]
joint_allps=[]
joint_alls=[]
    
reference, candidate, source, alignments = [], [], [], []
for src1, src1_len, src2,src2_len, src3, src3_len, tgt, tgt_len,tgtv, tgtv_len,tgtpv, tgtpv_len in testloader:

    if use_cuda:
        src1 = src1.cuda()
        src2 = src2.cuda()
        src3 = src3.cuda()
        tgtpv = tgtpv.cuda()
        src1_len = src1_len.cuda()
        src2_len = src2_len.cuda()
        src3_len = src3_len.cuda()
        tgtpv_len = tgtpv_len.cuda()

        samples,ssamples,vsamples,_ = model.sample(src1, src1_len, src2,src2_len, src3, src3_len,tgtpv, tgtpv_len)
        # get prediction sequence    
        for x,xv,xvv,y,yv,yvv in zip(samples,ssamples,vsamples, tgt[0],tgtv[0],tgtpv[0]):
            #each turn
            x=x.data.cpu()
            y=y.data.cpu()
            xt=x[0][:-1].tolist()
            preds.append(xt)
            svt=[]
            svt.extend(xt)
            for k in xv:
                svt.extend(k[0][:-1].tolist())
            joint_preds.append(svt)

            
            vvt=[]
            vvt.extend(svt)
            for k in xvv:
                for j in k:
                    vvt.extend(j[0][:-1].tolist())
            joint_allps.append(vvt)

            #print(joint_preds)
            label = []
            for l in y[1:].tolist():
                if l == 102:
                    break
                label.append(l)

            labels.append(label) 

            joint_label = []
            joint_label.extend(label)
            for k in yv[1:].tolist():
                if sum(k[1:])==0:
                    break
                else:
                    for l in k[1:]:
                        if l == 102:
                            break
                        joint_label.append(l)
            joint_labels.append(joint_label)
            
            joint_all=[]
            joint_all.extend(joint_label)
            for j in yvv[1:].tolist():
                for k in j[1:]:
                    if sum(k[1:])==0:
                        break
                    else:
                        for l in k[1:]:
                            if l == 102:
                                break
                            joint_all.append(l)
            joint_alls.append(joint_all)

# calculate acc
acc = []
jacc=[]
jaacc=[]
for p,l,jp,jl,jap,jal in zip(preds, labels,joint_preds,joint_labels,joint_allps,joint_alls):
    acc.append(p == l) 
    jacc.append(jp == jl)
    jaacc.append(jap == jal)
acc=sum(acc) / len(acc)
jacc=sum(jacc) / len(jacc)
jaacc=sum(jaacc) / len(jaacc)
print("slot_acc = {}\n".format(acc))      
print("joint_ds_acc = {}\n".format(jacc))
print("joint_all_acc = {}\n".format(jaacc))
  
