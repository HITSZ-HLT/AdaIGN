import argparse
import numpy as np
import datetime
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_use import IEMOCAPDataset, MELDDataset
from model_ada_use import  DialogueGNNModel_ada
from model import LSTMModel, GRUModel, DialogRNNModel, DialogueGNNModel
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support
from loss import FocalLoss, MaskedNLLLoss
from torch.nn.utils.rnn import pad_sequence
import math

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_class_weight(mu=1):
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}        
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False):
    path = '/data/tugeng/xt/MM-DFN (1)/data/meld/MELD_features_raw1.pkl'
    roberta_path ='/data/tugeng/xt/MM-DFN (1)/meld_features_roberta_xt.pkl'
    # trainset = MELDDataset(path,roberta_path,True)
    trainset = MELDDataset(path,roberta_path,'train')
    validset=MELDDataset(path,roberta_path,'valid')
    testset=MELDDataset(path,roberta_path,'test')
    # train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,#sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,#sampler=valid_sampler,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    # testset = MELDDataset(path,roberta_path,train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


# def get_IEMOCAP_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False):
#     path = '/data/tugeng/xt/myMM-DFN/data/iemocap/IEMOCAP_features.pkl'
#     roberta_path ='/data/tugeng/xt/myMM-DFN/data/iemocap/iemocap_features_roberta.pkl'
#     trainset = IEMOCAPDataset(path,roberta_path,'train')
#     validset=IEMOCAPDataset(path,roberta_path,'valid')
#     testset=IEMOCAPDataset(path,roberta_path,'test')
#     # trainset = IEMOCAPDataset(path=data_path)
#     # train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

#     train_loader = DataLoader(trainset,
#                               batch_size=batch_size,
#                               collate_fn=trainset.collate_fn,
#                               num_workers=num_workers,
#                               pin_memory=pin_memory)

#     valid_loader = DataLoader(validset,
#                               batch_size=batch_size,#   sampler=valid_sampler,
#                               collate_fn=validset.collate_fn,
#                               num_workers=num_workers,
#                               pin_memory=pin_memory)

#     # testset = IEMOCAPDataset(path=data_path, train=False)
#     test_loader = DataLoader(testset,
#                              batch_size=batch_size,
#                              collate_fn=testset.collate_fn,
#                              num_workers=num_workers,
#                              pin_memory=pin_memory)

#     return train_loader, valid_loader, test_loader
def get_IEMOCAP_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False):
    path = '../IEMOCAP_features.pkl'
    roberta_path ='../iemocap_features_roberta_xt.pkl'
    trainset = IEMOCAPDataset(path,roberta_path,'train')
    validset=IEMOCAPDataset(path,roberta_path,'valid')
    testset=IEMOCAPDataset(path,roberta_path,'test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,#sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,#sampler=valid_sampler,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    # testset = MELDDataset(path,roberta_path,train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

'''
def get_IEMOCAP_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=data_path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=data_path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader
'''

def train_or_eval_graph_model(folder,model, loss_f, dataloader, epoch, flag, is_policy,train_flag=False, optimizer_w=None,optimizer_a=None, cuda_flag=False, modals=None, target_names=None,
                              test_label=False, tensorboard=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []
    num_train_layers =200
    # print('num_train_layers:',num_train_layers)
    if cuda_flag: ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train_flag or optimizer_w != None
    if train_flag:
        model.train()
    else:
        model.eval()
    seed_everything(1137)
    
    modal_init = torch.load('../outputs (1)/iemocaprobmodal/mmdfn_base/model_27.pkl')
    modal_a=torch.load('../outputs (1)/iemocaprobmodala/mmdfn_base/model_27.pkl')
    modal_v=torch.load('../outputs (1)/iemocaprobmodalv/mmdfn_base/model_13.pkl')
    modal_l=torch.load('../outputs (1)/iemocaprobmodall/mmdfn_base/model_5.pkl')
    model_con_a=torch.load('../outputs (1)/iemocaprobcontexta/mmdfn_base/model_9.pkl')
    model_con_v=torch.load('../outputs (1)/iemocaprobcontextv/mmdfn_base/model_29.pkl')
    model_con_t=torch.load('../outputs (1)/iemocaprobcontextl/mmdfn_base/model_9.pkl')
    modal_init.eval()
    modal_a.eval()
    modal_v.eval()
    modal_l.eval()

    model_con_a.eval()
    model_con_v.eval()
    model_con_t.eval()
    
    for data in dataloader:
        a=0
        v=0
        c_a=0
        c_v=0
        c_l=0
        if train_flag:
            optimizer_w.zero_grad()
        # if folder=='IEMOCAP':
        #     textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        # else:
        textf1,textf2,textf3,textf4,visuf,acouf, qmask, umask, label =\
                [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        textf=[textf1,textf2,textf3,textf4]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        log_prob1, e_i, e_n, e_t, e_l = modal_init(textf, qmask, umask, lengths, acouf, visuf, True)
        pred_init = torch.argmax(log_prob1, 1)
        pred_init =[pred_init[:x] for i, x in enumerate(lengths)]
        pred_init=pad_sequence(pred_init)
        mask=torch.zeros_like(visuf)
        log_prob1, e_i, e_n, e_t, e_l = modal_v(textf, qmask, umask, lengths, acouf, mask,True)
        pred_maskv = torch.argmax(log_prob1, 1)
        pred_maskv =[pred_maskv[:x] for i, x in enumerate(lengths)]
        pred_maskv=pad_sequence(pred_maskv)
        mask=torch.zeros_like(acouf)
        log_prob1, e_i, e_n, e_t, e_l = modal_a(textf, qmask, umask, lengths, mask, visuf, True)
        pred_maska = torch.argmax(log_prob1, 1)
        pred_maska =[pred_maska[:x] for i, x in enumerate(lengths)]
        pred_maska=pad_sequence(pred_maska)
        if isinstance(textf,list):
            [t1,t2,t3,t4]=textf
            mask1=torch.zeros_like(t1)
            mask2=torch.zeros_like(t2)
            mask3=torch.zeros_like(t3)
            mask4=torch.zeros_like(t4)
            mask=[mask1,mask2,mask3,mask4]
        else:
            mask=torch.zeros_like(textf)
        log_prob1, e_i, e_n, e_t, e_l = modal_l(mask, qmask, umask, lengths, acouf, visuf, True)
        pred_maskt = torch.argmax(log_prob1, 1)
        pred_maskt =[pred_maskt[:x] for i, x in enumerate(lengths)]
        pred_maskt=pad_sequence(pred_maskt)
        position_a=(pred_init==pred_maska).nonzero()
            # mask_a=torch.zeros_like(acouf)
            # for i in position_a:
            #     mask_a[i[0],i[1],:]=1
            # acouf=acouf.mul(mask_a)
        position_v=(pred_init==pred_maskv).nonzero()
            # mask_v=torch.zeros_like(visuf)
            # for i in position_v:
            #     mask_v[i[0],i[1],:]=1
            # visuf=visuf.mul(mask_v)
            # if isinstance(textf,list):
        position_t=(pred_init==pred_maskt).nonzero()
        # mask_t=torch.zeros_like(textf[0])
        # for i in position_t:
        #             mask_t[i[0],i[1],:]=1
        #         textf[0]=textf[0].mul(mask_t)
        #         textf[1]=textf[1].mul(mask_t)
        #         textf[2]=textf[2].mul(mask_t)
        #         textf[3]=textf[3].mul(mask_t)

        #         mask_t=torch.zeros_like(textf)
        #         for i in position_t:
        #             mask_t[i[0],i[1],:]=1
        #         textf=textf.mul(mask_t)
        log_prob1, e_i, e_n, e_t, e_l = model_con_a(textf, qmask, umask, lengths, acouf, visuf, True,"acouf")
        pred_con_a = torch.argmax(log_prob1, 1)
        pred_con_a =[pred_con_a[:x] for i, x in enumerate(lengths)]
        pred_con_a=pad_sequence(pred_con_a)
        log_prob1, e_i, e_n, e_t, e_l = model_con_v(textf, qmask, umask, lengths, acouf, visuf, True,"visuf")
        pred_con_v = torch.argmax(log_prob1, 1)
        pred_con_v =[pred_con_v[:x] for i, x in enumerate(lengths)]
        pred_con_v=pad_sequence(pred_con_v)
        log_prob1, e_i, e_n, e_t, e_l = model_con_t(textf, qmask, umask, lengths, acouf, visuf, True,"textf")
        pred_con_t = torch.argmax(log_prob1, 1)
        pred_con_t =[pred_con_t[:x] for i, x in enumerate(lengths)]
        pred_con_t=pad_sequence(pred_con_t)
        position_con_a=(pred_init==pred_con_a).nonzero()
        position_con_v=(pred_init==pred_con_v).nonzero()
        position_con_t=(pred_init==pred_con_t).nonzero()
        a+=position_a.size(0)
        v+=position_v.size(0)
        c_a+=position_con_a.size(0)
        c_v+=position_con_v.size(0)
        c_l+=position_con_t.size(0)
        # log_prob, [ploss_h,ploss_s] = model(textf, qmask, umask, lengths, acouf, visuf, test_label,num_train_layers, flag, train_flag,position_con_a,position_con_v,position_con_t,position_a,position_t,position_v)
        log_prob, [ploss_i,ploss_h,ploss_s,ploss_n] = model(textf, qmask, umask, lengths, acouf, visuf, test_label,num_train_layers, flag, train_flag,position_con_a,position_con_v,position_con_t,position_a,None,position_v)    
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        umask = torch.cat([umask[j][:lengths[j]] for j in range(len(umask))])
        if folder=='IEMOCAP':
            loss = loss_f(log_prob, label)
        else:
            loss = loss_f(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            if flag == 'update_w':
                optimizer_w.zero_grad()
                model.fix_alpha()
                    #for name,param in model.dialog_rnn_f.named_parameters():
                    #    print('name:',name,'requires_grad:',param.requires_grad,'gard value:',param.grad)
                loss.backward()
                if args.tensorboard:
                    for param in model.named_parameters():
                        writer.add_histogram(param[0], param[1].grad, epoch)
                optimizer_w.step()
            elif flag == 'update_alpha':
                optimizer_a.zero_grad()
                
                model.fix_w()
                    #for name,param in model.dialog_rnn_f.named_parameters():
                    #    print('name:',name,'requires_grad:',param.requires_grad,'gard value:',param.grad)
            
                ploss = args.hamming_w * ploss_h + args.sparse_w * ploss_s+args.importance_w * ploss_i + args.not_importance_w * ploss_n
                    # print('ploss:',ploss)
                    # print('ploss_h:',ploss_h )
                    # print('ploss_s:',ploss_s )
                (loss + ploss).backward()
                optimizer_a.step()
            else:
                raise ValueError('flag %s is not recognized' % flag)
        # print("a:%d",a)
        # print("v:%d",v)
        # print("c_a:%d",c_a)
        # print("c_v:%d",c_v)
        # print("c_l:%d",c_l)
    
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return [], [], float('nan'), float('nan'), [], [], float('nan'), []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_each = metrics.classification_report(labels, preds, target_names=target_names, digits=4)
    all_acc = ["ACC"]
    for i in range(len(target_names)):
        all_acc.append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    return all_each, all_acc, avg_loss, avg_accuracy, labels, preds, avg_fscore, [vids, ei, et, en, el]

def train_and_print(e, optimizer, bests, flag,is_policy, ToVal=True, folder=None):
    
    #model, loss_f, dataloader, e, flag, is_policy,train_flag=False, optimizer=None,optimizer_a=None, cuda_flag=False, modals=None, target_names=None,
    #                          test_label=False, tensorboard=False
    
    
    _, _, train_loss, train_acc, _, _, train_fscore, _ = train_or_eval_graph_model(folder,model,loss_f,train_loader,e,flag, is_policy,True,optimizer_w,optimizer_a,cuda_flag,args.modals,target_names)

    [best_loss, best_label, best_pred,  best_fscore] = bests
    
    results_loss=[]
    results_acc=[]
    results_fscore=[]
    results_label=[]
    results_predict=[]
   
    if ToVal == True:
        _, _, valid_loss, valid_acc, _, _, valid_fscore, _ = train_or_eval_graph_model(folder,model,loss_f,valid_loader,e,flag, is_policy,False,optimizer_w,optimizer_a,cuda_flag,args.modals,target_names)
        if e < 13:
            sampling = 1
        else:
            sampling = args.sampling
        for kk in range(sampling):
            #print('sampling time:',kk)
            all_each, all_acc, test_loss, test_acc, test_label, test_pred, test_fscore, attentions = train_or_eval_graph_model(folder,model,loss_f,test_loader,e,flag, is_policy,False,optimizer_w,optimizer_a,cuda_flag,args.modals,target_names)
            #print('test_fscore:',test_fscore)
            results_acc.append(test_acc)
            results_loss.append(test_loss)
            results_fscore.append(test_fscore)
            results_label.append(test_label)
            results_predict.append(test_pred)
            

            if best_fscore == None or best_fscore < test_fscore:
                best_loss, best_label, best_pred,  best_attn, best_fscore =\
                        test_loss, test_label, test_pred,  attentions, test_fscore
                import os

                save_dir = args.save_model_dir
                # if not os.path.isdir(save_dir): os.makedirs(save_dir)
                # torch.save(model, os.path.join(save_dir, 'model_' + str(e) + '.pkl'))
                
               
        
        test_fscore = round(np.asarray(results_fscore, dtype=np.float).max().squeeze(),4)
        avg = round(np.asarray(results_fscore, dtype=np.float).mean().squeeze(),2)
        best_num = np.asarray(results_fscore, dtype=np.float).argmax()
        test_loss = results_loss[best_num]
        test_label = results_label[best_num]
        test_pred = results_predict[best_num]
    
        test_acc = results_acc[best_num]
        avg_acc = round(np.asarray(results_acc, dtype=np.float).mean().squeeze(),2)

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss',test_acc/test_loss,e)
            writer.add_scalar('train: accuracy/loss',train_acc/train_loss,e)       
        print('ep{} train_loss {} train_acc {} train_fscore {} val_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} avg_acc {} avg {} time {}'.\
                format(e, train_loss, train_acc, train_fscore, valid_acc, valid_fscore,\
                        test_loss, test_acc, test_fscore, avg_acc, avg, round(time.time()-start_time,2)))
        print('    ')
    return [best_loss, best_label, best_pred, best_fscore]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--data_dir', type=str, default='/data/tugeng/xt/myMM-DFN/data/iemocap/IEMOCAP_features.pkl', help='dataset dir')

    parser.add_argument('--multi_modal', action='store_true', default=True, help='whether to use multimodal information')

    parser.add_argument('--modals', default='avl', help='modals to fusion: avl')

    parser.add_argument('--mm_fusion_mthd', default='concat_subsequently',
                        help='method to use multimodal information: mfn, concat, gated, concat_subsequently,mfn_only,tfn_only,lmf_only')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--base_model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU/None')

    parser.add_argument('--graph_model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--graph_type', default='GDF', help='relation/GCN3/DeepGCN/GF/GF2/GDF')

    parser.add_argument('--graph_construct', default='direct', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--nodal_attention', action='store_true', default=True, help='whether to use nodal attention in graph model')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--use_residue', action='store_true', default=True, help='whether to use residue information or not')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--active_listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--use_crn_speaker', action='store_true', default=False, help='whether to use use crn_speaker embedding')

    parser.add_argument('--speaker_weights', type=str, default='3-0-1', help='speaker weight 0-0-0')

    parser.add_argument('--use_speaker', action='store_true', default=False, help='whether to use speaker embedding')

    parser.add_argument('--reason_flag', action='store_true', default=False, help='reason flag')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--valid_rate', type=float, default=0.0, metavar='valid_rate', help='valid rate, 0.0/0.1')

    parser.add_argument('--modal_weight', type=float, default=1.0, help='modal weight 1/0.7')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=16, help='Deep_GCN_nlayers')

    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_a', type=float, default=2e-2, metavar='LR_a',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec_dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha 0.1/0.2')

    parser.add_argument('--lamda', type=float, default=0.5, help='eta 0.5/0')

    parser.add_argument('--gamma', type=float, default=0.5, help='gamma 0.5/1/2')

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--loss', default="FocalLoss", help='loss function: FocalLoss/NLLLoss')

    parser.add_argument('--class_weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--save_model_dir', type=str, default='/data/tugeng/xt/myMM-DFN/outputs/iemocap_demo1137/', help='saved model dir')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--test_label', action='store_true', default=False, help='whether do test only')

    parser.add_argument('--load_model', type=str, default='/data10T/tugeng/xt/AdaGIN(result)/outputs/iemocap_demo/model_46.pkl', help='trained model dir')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='E',help='number of epochs')
    parser.add_argument('--temp', type=float, default=5.0, metavar='temp',help='init temperature')   
    parser.add_argument('--ploss_reverse', action='store_true', default=False,
                        help='ploss_reverse')  
    parser.add_argument('--hamming_w', type=float, default=0.6, metavar='hamming',
                        help='hamming loss weight')       
    parser.add_argument('--sparse_w', type=float, default=0.1, metavar='sparse',
                        help='sparse loss weight') 
    parser.add_argument('--importance_w', type=float, default=0.9, metavar='hamming',
                        help='hamming loss weight')       
    parser.add_argument('--not_importance_w', type=float, default=0.2, metavar='sparse',
                        help='sparse loss weight') 
    parser.add_argument('--temp_decay', type=float, default=0.965, metavar='tempdacay',
                        help='temperature decay')  
    parser.add_argument('--sampling', type=int, default=1,
                        help='number of samplings') 
                    
    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)
   
    name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + str(args.Deep_GCN_nlayers) + '_' + args.dataset

    cuda_flag = torch.cuda.is_available() and not args.no_cuda

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = feat2dim['textCNN'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_text']

    
    D_m = D_text
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    n_speakers, n_classes, class_weights, target_names = -1, -1, None, None
    if args.dataset == 'IEMOCAP':
        n_speakers, n_classes = 2, 6
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        class_weights = torch.FloatTensor([1 / 0.086747,
                                           1 / 0.144406,
                                           1 / 0.227883,
                                           1 / 0.160585,
                                           1 / 0.127711,
                                           1 / 0.252668])
    if args.dataset == 'MELD':
        n_speakers, n_classes = 9, 7

        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
        class_weights = torch.FloatTensor([1.0 / 0.466750766,
                                           1.0 / 0.122094071,
                                           1.0 / 0.027752748,
                                           1.0 / 0.071544422,
                                           1.0 / 0.171742656,
                                           1.0 / 0.026401153,
                                           1.0 / 0.113714183])


    

   
    
    seed_everything(1137)	# 设置随机种子，每次搜索设置不同的种子，若种子固定，那每次选取的超参都是一样的
        
    model = DialogueGNNModel_ada(args.base_model,args.temp,
                                    D_m, D_g, D_p, D_e, D_h, D_a,args.batch_size, graph_h,
                                    n_speakers=n_speakers,
                                    max_seq_len=200,
                                    window_past=args.windowp,
                                    window_future=args.windowf,
                                    n_classes=n_classes,
                                    listener_state=args.active_listener,
                                    context_attention=args.attention,
                                    dropout=args.dropout,
                                    nodal_attention=args.nodal_attention,
                                    no_cuda=not cuda_flag,
                                    graph_type=args.graph_type,
                                    use_topic=args.use_topic,
                                    alpha=args.alpha,
                                    lamda=args.lamda,
                                    multiheads=args.multiheads,
                                    graph_construct=args.graph_construct,
                                    use_GCN=args.use_gcn,
                                    use_residue=args.use_residue,
                                    D_m_v=D_visual,
                                    D_m_a=D_audio,
                                    modals=args.modals,
                                    att_type=args.mm_fusion_mthd,
                                    av_using_lstm=args.av_using_lstm,
                                    Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                    dataset=args.dataset,
                                    use_speaker=args.use_speaker,
                                    use_modal=args.use_modal,
                                    reason_flag=args.reason_flag,
                                    multi_modal=args.multi_modal,
                                    use_crn_speaker=args.use_crn_speaker,
                                    speaker_weights=args.speaker_weights,
                                    modal_weight=args.modal_weight,
                                    temp_decay=args.temp_decay)

    name = 'MM-DFN'
    print('{} with {} as base model'.format(name, args.base_model))
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(modals))

    if cuda_flag:
        # torch.cuda.set_device(0)
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    if args.loss == 'FocalLoss' and args.graph_model:
        # FocalLoss
        loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None)
    else:
        # NLLLoss
        loss_f = nn.NLLLoss(class_weights if args.class_weight else None) if args.graph_model \
            else MaskedNLLLoss(class_weights if args.class_weight else None)
    # if args.dataset == 'MELD':
    #     if args.class_weight:
    #         if args.mu > 0:
    #             loss_weights = torch.FloatTensor(create_class_weight(args.mu))
    #         else:   
    #             loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 
    #                       0.84847735, 5.42461417, 1.21859721])
    #             loss_f  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    #     else:
    #         loss_f = MaskedNLLLoss()
    # else:loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None)
    params_w = model.network_parameters() 
    params_a = model.arch_parameters() 
    optimizer_w = optim.Adam(params_w,
                        lr=args.lr,
                        weight_decay=args.l2)
    optimizer_a=optim.Adam(params_a, lr=args.lr_a, weight_decay=5*1e-4)

    if args.dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(data_path=args.data_dir,
                                                                valid_rate=args.valid_rate,
                                                                batch_size=batch_size,
                                                                num_workers=0)
    elif args.dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(data_path=args.data_dir,
                                                                    valid_rate=args.valid_rate,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    else:
        train_loader, valid_loader, test_loader = None, None, None
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred= None, None, None, None
    bests = [best_loss, best_label, best_pred, best_fscore]
    all_fscore, all_acc, all_loss = [], [], []
    model.fix_alpha()
    model.free_w()
    lr=args.lr
    lr_a=args.lr_a
    flag = 'update_w'
    current_iter_w, current_iter_a = 0, 0
    folder = 'experiments/MELD/%s_adasampling' % (time.ctime()[-20:-17]+time.ctime()[-16:-14]+'_'+time.ctime()[-13:-11]+time.ctime()[-10:-8]+time.ctime()[-7:-5])
    for e in range(n_epochs):
        start_time = time.time()
        if e < args.warmup_epochs:
            print('warming up')
            bests = train_and_print(e, optimizer_w, bests, flag, is_policy=False, ToVal=True, folder=args.dataset) 
        
    
        else:
            if e == args.warmup_epochs:
                    #optimizer = optim.Adam(params_w + list(caps_Layer.parameters()), lr=0.0003, weight_decay=args.l2)
                    print('warm up ending')
                    model.fix_alpha()
            
            if flag == 'update_w':
                    print('updating w')
                    bests = train_and_print(e, optimizer_w, bests, flag,is_policy=True, ToVal=True, folder=args.dataset) 
                    flag = 'update_alpha'
                    model.fix_w()
                
                    model.free_alpha()
            
            elif flag == 'update_alpha':
                    print('updating alpha')
                    bests = train_and_print(e, optimizer_w, bests, flag, is_policy=True,ToVal=False,folder=args.dataset) 
                    flag = 'update_w'
                    model.fix_alpha()
                    model.free_w()
                    model.decay_temperature()
                    dists1,dist2 = model.get_policy_prob(batch_size) #输出modal%d_logits经过了softmax之后的distribution
                    # print(np.concatenate(dists1, axis=-1)[-6:])
                
        if (e+1) % 25 == 0 or (e+1) % 40 == 0:
            lr = 0.6*lr
            optimizer_w = optim.Adam(params_w , lr=lr, weight_decay=args.l2)
            print('--------------------decrease lr---------------------')
            lr_a = 0.8*lr_a
                #optimizer_a = optim.Adam(params_a, lr=lr_a, weight_decay=args.l2)
            optimizer_a = optim.RMSprop(params_a, lr=lr_a, weight_decay=args.l2, momentum=0.95)
            print('--------------------decrease lr_a---------------------')
                
    if args.tensorboard:
        writer.close()

    [best_loss, best_label, best_pred,  best_fscore] = bests
    print('Test performance..')
    print('Fscore {} accuracy {}'.format(best_fscore,round(accuracy_score(best_label,best_pred)*100,2)))
    print(classification_report(best_label,best_pred,digits=4))
    print(confusion_matrix(best_label,best_pred))

    
        