import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx, ImageList_idx_aug, ImageList_idx_fix
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import heapq
from numpy import linalg as LA
from loss import CrossEntropyLabelSmooth

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load_oh(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = 64

    path_Art = '/home/ts/projects/SYJ/ACM_contiual_sfda/data_shot/office-home/Art_list.txt' 
    path_Clipart = '/home/ts/projects/SYJ/ACM_contiual_sfda/data_shot/office-home/Clipart_list.txt' 
    path_Product = '/home/ts/projects/SYJ/ACM_contiual_sfda/data_shot/office-home/Product_list.txt' 
    path_RealWord = '/home/ts/projects/SYJ/ACM_contiual_sfda/data_shot/office-home/RealWorld_list.txt' 
    
    txt_tar_a = open(path_Art).readlines()
    txt_tar_c = open(path_Clipart).readlines()
    txt_tar_p = open(path_Product).readlines()
    txt_tar_r = open(path_RealWord).readlines()

    tr_txt_a = torch.load(osp.join(args.data_sp_src, "tr_txt_a.pt")) 
    te_txt_a = torch.load(osp.join(args.data_sp_src, "te_txt_a.pt"))
    tr_txt_c = torch.load(osp.join(args.data_sp_src, "tr_txt_c.pt")) 
    te_txt_c = torch.load(osp.join(args.data_sp_src, "te_txt_c.pt"))
    tr_txt_p = torch.load(osp.join(args.data_sp_src, "tr_txt_p.pt")) 
    te_txt_p = torch.load(osp.join(args.data_sp_src, "te_txt_p.pt")) 
    tr_txt_r = torch.load(osp.join(args.data_sp_src, "tr_txt_r.pt")) 
    te_txt_r = torch.load(osp.join(args.data_sp_src, "te_txt_r.pt")) 

    dsets["Art_te"] = ImageList(te_txt_a, transform=image_test())
    dset_loaders["Art_te"] = DataLoader(dsets["Art_te"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["Art_tr"] = ImageList(tr_txt_a, transform=image_train())
    dset_loaders["Art_tr"] = DataLoader(dsets["Art_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["Clipart_te"] = ImageList(te_txt_c, transform=image_test())
    dset_loaders["Clipart_te"] = DataLoader(dsets["Clipart_te"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["Clipart_tr"] = ImageList(tr_txt_c, transform=image_train())
    dset_loaders["Clipart_tr"] = DataLoader(dsets["Clipart_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["Product_te"] = ImageList(te_txt_p, transform=image_test())
    dset_loaders["Product_te"] = DataLoader(dsets["Product_te"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["Product_tr"] = ImageList(tr_txt_p, transform=image_train())
    dset_loaders["Product_tr"] = DataLoader(dsets["Product_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["RealWorld_te"] = ImageList(te_txt_r, transform=image_test())
    dset_loaders["RealWorld_te"] = DataLoader(dsets["RealWorld_te"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["RealWorld_tr"] = ImageList(tr_txt_r, transform=image_train())
    dset_loaders["RealWorld_tr"] = DataLoader(dsets["RealWorld_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["Art_test"] = ImageList(txt_tar_a, transform=image_test())
    dset_loaders["Art_test"] = DataLoader(dsets["Art_test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["Art_target"] = ImageList_idx(txt_tar_a, transform=image_train())
    dset_loaders["Art_target"] = DataLoader(dsets["Art_target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["Art_test_aug"] = ImageList_idx_aug(txt_tar_a, transform=image_test())
    dset_loaders["Art_test_aug"] = DataLoader(dsets["Art_test_aug"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    
    dsets["Clipart_test"] = ImageList(txt_tar_c, transform=image_test())
    dset_loaders["Clipart_test"] = DataLoader(dsets["Clipart_test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["Clipart_target"] = ImageList_idx(txt_tar_c, transform=image_train())
    dset_loaders["Clipart_target"] = DataLoader(dsets["Clipart_target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["Clipart_test_aug"] = ImageList_idx_aug(txt_tar_c, transform=image_test())
    dset_loaders["Clipart_test_aug"] = DataLoader(dsets["Clipart_test_aug"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    dsets["Product_test"] = ImageList(txt_tar_p, transform=image_test())
    dset_loaders["Product_test"] = DataLoader(dsets["Product_test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["Product_target"] = ImageList_idx(txt_tar_p, transform=image_train())
    dset_loaders["Product_target"] = DataLoader(dsets["Product_target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["Product_test_aug"] = ImageList_idx_aug(txt_tar_p, transform=image_test())
    dset_loaders["Product_test_aug"] = DataLoader(dsets["Product_test_aug"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    dsets["RealWorld_test"] = ImageList(txt_tar_r, transform=image_test())
    dset_loaders["RealWorld_test"] = DataLoader(dsets["RealWorld_test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["RealWorld_target"] = ImageList_idx(txt_tar_r, transform=image_train())
    dset_loaders["RealWorld_target"] = DataLoader(dsets["RealWorld_target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["RealWorld_test_aug"] = ImageList_idx_aug(txt_tar_r, transform=image_test())
    dset_loaders["RealWorld_test_aug"] = DataLoader(dsets["RealWorld_test_aug"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders


def prepare_data_oh(args): 
    ## prepare data
    path_Art = '/home/ts/projects/SYJ/ACM_contiual_sfda/data_shot/office-home/Art_list.txt' 
    path_Clipart = '/home/ts/projects/SYJ/ACM_contiual_sfda/data_shot/office-home/Clipart_list.txt' 
    path_Product = '/home/ts/projects/SYJ/ACM_contiual_sfda/data_shot/office-home/Product_list.txt' 
    path_RealWord = '/home/ts/projects/SYJ/ACM_contiual_sfda/data_shot/office-home/RealWorld_list.txt' 
    
    txt_src_a = open(path_Art).readlines()
    txt_src_c = open(path_Clipart).readlines()
    txt_src_p = open(path_Product).readlines()
    txt_src_r = open(path_RealWord).readlines()

    dsize_a = len(txt_src_a)
    tr_size_a = int(0.9*dsize_a)
    print(dsize_a, tr_size_a, dsize_a - tr_size_a)
    tr_txt_a, te_txt_a = torch.utils.data.random_split(txt_src_a, [tr_size_a, dsize_a - tr_size_a])
    torch.save(tr_txt_a, osp.join(args.data_sp_src, 'tr_txt_a.pt'))
    torch.save(te_txt_a, osp.join(args.data_sp_src, 'te_txt_a.pt'))

    dsize_c = len(txt_src_c)
    tr_size_c = int(0.9*dsize_c)
    print(dsize_c, tr_size_c, dsize_c - tr_size_c)
    tr_txt_c, te_txt_c = torch.utils.data.random_split(txt_src_c, [tr_size_c, dsize_c - tr_size_c])
    torch.save(tr_txt_c, osp.join(args.data_sp_src, 'tr_txt_c.pt'))
    torch.save(te_txt_c, osp.join(args.data_sp_src, 'te_txt_c.pt'))

    dsize_p = len(txt_src_p)
    tr_size_p = int(0.9*dsize_p)
    print(dsize_p, tr_size_p, dsize_p - tr_size_p)
    tr_txt_p, te_txt_p = torch.utils.data.random_split(txt_src_p, [tr_size_p, dsize_p - tr_size_p])
    torch.save(tr_txt_p, osp.join(args.data_sp_src, 'tr_txt_p.pt'))
    torch.save(te_txt_p, osp.join(args.data_sp_src, 'te_txt_p.pt'))

    dsize_r = len(txt_src_r)
    tr_size_r = int(0.9*dsize_r)
    print(dsize_r, tr_size_r, dsize_r - tr_size_r)
    tr_txt_r, te_txt_r = torch.utils.data.random_split(txt_src_r, [tr_size_r, dsize_r - tr_size_r])
    torch.save(tr_txt_r, osp.join(args.data_sp_src, 'tr_txt_r.pt'))
    torch.save(te_txt_r, osp.join(args.data_sp_src, 'te_txt_r.pt'))

    print("data office-home has splited")
    return 0


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def train_source(loader_tr, loader_te, args):
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  
    elif args.net == 'vit':
        netF = network.ViT().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch_src * len(loader_tr)
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(loader_tr)
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(loader_te, netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(loader_te, netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file_src.write(log_str + '\n')
            args.out_file_src.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()
                
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC


def train_target(loader_tar, loader_tst_aug, loader_tst, loader_ss,  netF, netB, netC, args):
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch_tgt * len(loader_tar)
    interval_iter = max_iter // args.interval
    iter_num = 0

    pred_lab_near_bank = []
    opts_aug_bank = []
    feas_all_aug_bank = []
    feas_all_near_bank = []
    acc_all_ =[]

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(loader_tar)
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            pred_near, opts_aug, feas_FC = obtain_label(loader_tst_aug, netF, netB, netC, args)
            feas_all_aug = feas_FC[0]
            feas_all_near = feas_FC[1]

            opts_aug_bank.append(opts_aug)
            pred_lab_near_bank.append(pred_near)
            
            feas_all_aug = torch.from_numpy(feas_all_aug)
            feas_all_aug_bank.append(feas_all_aug)

            feas_all_near = torch.from_numpy(feas_all_near)
            feas_all_near_bank.append(feas_all_near)
            
            netF.train()
            netB.train()
            netC.train()

        inputs_test = inputs_test.cuda()
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)


        features_F_self_B = netB(netF(inputs_test))
        outputs_test = netC(features_F_self_B)

        features_B_self = features_F_self_B.clone().detach()
        mem_outputs_new = obtain_outputs_his_ls(features_B_self, feas_all_aug_bank, opts_aug_bank, tar_idx)
        mem_label_soft = obtain_label_his_ls(features_B_self, feas_all_near_bank, pred_lab_near_bank, tar_idx, args)
        mem_label_soft = mem_label_soft.cuda()

        outputs_t = mem_outputs_new.cuda()
        outputs_s = outputs_test

        temperature = 1


        if args.cls_par > 0:
            log_probs = nn.LogSoftmax(dim=1)(outputs_test)
            targets = mem_label_soft
            loss_soft = (- targets * log_probs).sum(dim=1)
            classifier_loss = loss_soft.mean() 
            classifier_loss *= args.cls_par

            KD_loss = - 1 * (F.softmax(outputs_t / temperature, 1).detach() * \
                F.log_softmax(outputs_s / temperature, 1)).sum() / inputs_test.size()[0]

            classifier_loss = classifier_loss + 1.3*KD_loss

            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss = KD_loss
        else:                                                                                                                                                                                                                  
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test) 
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= 1 * gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
            
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                # acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                # log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
                acct, acc_list_t = cal_acc(loader_tst, netF, netB, netC, True)
                accs, acc_list_s = cal_acc(loader_ss, netF, netB, netC, True)
                acc_all = [accs, acct]
                log_str = 'Task: {}, Iter:{}/{}; Accuracy on source = {:.2f}%. Accuracy on target = {:.2f}%'.format(args.name, iter_num, max_iter, accs, acct) + '\n' + acc_list_s +'\n' + acc_list_t
            else:
                # acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                # log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
                acct, _ = cal_acc(loader_tst, netF, netB, netC, False)
                accs, _ = cal_acc(loader_ss, netF, netB, netC, False)
                acc_all = [accs, acct]
                log_str = 'Task: {}, Iter:{}/{}; Accuracy on source = {:.2f}%. Accuracy on target = {:.2f}%'.format(args.name, iter_num, max_iter, accs, acct)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()
            netC.train()
            acc_all_.append(acc_all)

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        np.save(osp.join(args.output_dir, "acc_all_30" + ".npy"), acc_all_)
        
    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0][0]
            inputs_aug = data[0][1][0]
            labels = data[1]
            inputs = inputs.cuda()
            feas_F = netF(inputs)
            feas = netB(feas_F)
            outputs = netC(feas)
            inputs_aug = inputs_aug.cuda()
            feas_F_aug = netF(inputs_aug)
            feas_aug = netB(feas_F_aug)
            outputs_aug = netC(feas_aug)
            if start_test:
                all_fea = feas.float().cpu()
                all_fea_aug = feas_aug.float().cpu()
                all_output = outputs.float().cpu()
                all_output_aug = outputs_aug.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)          # 498*256
                all_output = torch.cat((all_output, outputs.float().cpu()), 0) # 498*31
                all_output_aug = torch.cat((all_output_aug, outputs_aug.float().cpu()), 0) # 498*31
                all_label = torch.cat((all_label, labels.float()), 0)          # 498
                all_fea_aug = torch.cat((all_fea_aug, feas_aug.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    all_output_aug = nn.Softmax(dim=1)(all_output_aug)

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)
    _, predict_aug = torch.max(all_output_aug, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    accuracy_aug = torch.sum(torch.squeeze(predict_aug).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea_aug = torch.cat((all_fea_aug, torch.ones(all_fea_aug.size(0), 1)), 1)
        all_fea_aug = (all_fea_aug.t() / torch.norm(all_fea_aug, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    all_fea_aug = all_fea_aug.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    xx = np.eye(K)[predict]
    cls_count = xx.sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea) 
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy_shot = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    log_str_aug = 'Accuracy_shot_aug = {:.2f}% -> {:.2f}%'.format(accuracy_aug * 100, acc * 100)

    pred_label_near, all_feas_near = obtain_near_label(all_fea, pred_label)
    feas_re = (all_fea_aug, all_feas_near)
    acc_near = np.sum(pred_label_near == all_label.float().numpy()) / len(all_fea)
    log_near = 'Accuracy_near = {:.2f}%'.format(acc_near * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    print(log_str_aug + '\n')
    print(log_near + '\n')
    return  pred_label_near.astype('int'), all_output_aug, feas_re


def obtain_near_label(feas, label_old):
    ln_sam = feas.shape[0]
    idx_row = np.array(range(ln_sam))
    dd_fea = np.dot(feas, feas.T)
    dd_fea[idx_row, idx_row] = -10000
    idx_col_max = dd_fea.argmax(axis=1)
    near_label = label_old[idx_col_max]
    feas_near = feas[idx_col_max]
    return near_label, feas_near


def obtain_outputs_his_ls(features_B, feas_all_aug_bank, outputs_bank, tar_idx):
    
    num_nn = len(outputs_bank)
    outputs_new = np.zeros_like(outputs_bank[0][tar_idx])
    val_dd_arr = obtain_sim(features_B, feas_all_aug_bank, tar_idx)  
    for i in range(num_nn):
        outputs_i = outputs_bank[i][tar_idx].numpy()
        decay_val = pow(0.8, (num_nn-i-1))
        wts_i = val_dd_arr[:, i][:, None]
        outputs_new += outputs_i * decay_val* wts_i
    outputs_new_ts = torch.from_numpy(outputs_new)
    outputs_new_soft = (outputs_new_ts.T / torch.max(outputs_new_ts, 1)[0]).T
    outputs_new_norm = outputs_new_soft
    # outputs_t = torch.tensor(mem_outputs_new).cuda()
    return outputs_new_norm


def obtain_sim(feas, feas_bank, tar_idx):
    val_arr = []
    num_nn = len(feas_bank)
    feas = feas.clone().detach().cpu()
    feas = torch.cat((feas, torch.ones(feas.size(0), 1)), 1)
    feas = (feas.t() / torch.norm(feas, p=2, dim=1)).t()
    for i in range(num_nn):
        features_B_i = feas_bank[i]
        features_B_batch_i = features_B_i[tar_idx]
        
        sim_feas_mat = torch.mm(feas, features_B_batch_i.T)
        ln_sam = feas.shape[0]
        idx_row = np.array(range(ln_sam))
        sim_feas = sim_feas_mat[idx_row, idx_row]
        sim_feas_ = sim_feas.numpy()
        val_arr.append(sim_feas_)
    val_dd_arr = np.vstack(tuple(val_arr)).T
    return val_dd_arr


def obtain_label_his_ls(features_B, feas_all_near_bank, pred_labs_near_bank, tar_idx, args):
    num_nn = len(pred_labs_near_bank)
    pred_label = np.zeros_like(np.eye(args.class_num)[pred_labs_near_bank[0][tar_idx]])
    val_dd_arr = obtain_sim(features_B, feas_all_near_bank, tar_idx)  
    for i in range(num_nn):
        pred_label_i = pred_labs_near_bank[i][tar_idx]
        one_hot_i = np.eye(args.class_num)[pred_label_i]
        decay_val = pow(0.8, num_nn-i-1)
        wts_i = val_dd_arr[:, i][:, None]
        pred_label += one_hot_i * wts_i*decay_val
    pred_labels = pred_label.astype('int')
    pred_labels = torch.from_numpy(pred_labels)
    return pred_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PKMSM')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch_src', type=int, default=100, help="max iterations")
    parser.add_argument('--max_epoch_tgt', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='vit', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.6)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--kd_par', type=float, default=1.3)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--smooth', type=float, default=0.1)  

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='./ckpspdtcda/target_pdtcda_tem/')
    parser.add_argument('--output_data', type=str, default='./data_sp_1/')
    # parser.add_argument('--output_src', type=str, default='./ckpskdr2/source_tt/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("---------------------------target_hisnei_new_zengq_04_10------------------------------")

    folder = './data_shot/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    # args.s_dset_path_test = folder + args.dset + '/' + names[args.s] + '_test.txt'
    # args.s_dset_path_train = folder + args.dset + '/' + names[args.s] + '_train.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.output_dir_src = osp.join(args.output, args.dset, names[args.s][0].upper())
    args.data_sp_src = osp.join(args.output_data, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()

    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    if not osp.exists(args.data_sp_src):
        os.system('mkdir -p ' + args.data_sp_src)
    if not osp.exists(args.data_sp_src):
        os.mkdir(args.data_sp_src)

    prepare_data_oh(args)

    dset_loaders = data_load_oh(args)

    if not osp.exists(osp.join(args.output_dir_src + '/source_F.pt')):
        args.out_file_src = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
        args.out_file_src.write(print_args(args)+'\n')
        args.out_file_src.flush()
        loader_tr_ = dset_loaders[str(names[args.s]) + '_tr']
        loader_te_ = dset_loaders[str(names[args.s]) + '_te']
        netF, netB, netC = train_source(loader_tr_, loader_te_, args)

        args.out_file_src = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t = i
            args.name = names[args.s][0].upper() + names[args.t][0].upper()
            acc, _ = cal_acc(dset_loaders[str(names[args.t]) + '_test'], netF, netB, netC, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)
            args.out_file_src.write(log_str)
            args.out_file_src.flush()
            print(log_str)
    else:
        if args.net[0:3] == 'res':
            netF = network.ResBase(res_name=args.net).cuda()
        elif args.net[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=args.net).cuda() 
        elif args.net == 'vit':
            netF = network.ViT().cuda() 
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
        netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
        modelpath = args.output_dir_src + '/source_F.pt'   
        netF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + '/source_B.pt'
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + '/source_C.pt'
        netC.load_state_dict(torch.load(modelpath))
        netF.eval()
        netB.eval()
        netC.eval()
        acc_s_te, _ = cal_acc(dset_loaders[str(names[args.s]) + '_te'], netF, netB, netC, False)
        print(acc_s_te)
    
    sss = args.s

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.output_dir = osp.join(args.output, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper() 

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()

        start =True
        if start:
            loader_tgt = dset_loaders[str(names[args.t]) + '_target']
            loader_tet_aug = dset_loaders[str(names[args.t]) + '_test_aug']
            loader_tet = dset_loaders[str(names[args.t]) + '_test']
            loader_ss = dset_loaders[str(names[args.s]) + '_te']
            netF, netB, netC = train_target(loader_tgt, loader_tet_aug, loader_tet, loader_ss, netF, netB, netC, args)
            for j in range(len(names)):
                args.s = j
                if args.s == sss:
                    continue
                acc, _ = cal_acc(dset_loaders[str(names[args.s]) + '_test'], netF, netB, netC, False)
                log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(names[args.t][0].upper(), names[args.s][0].upper(), acc)
                args.out_file.write(log_str)
                print(log_str)
            start=False
            args.s = sss
        else:
            loader_tgt = dset_loaders[str(names[args.t]) + '_target']
            loader_tet_aug = dset_loaders[str(names[args.t]) + '_test_aug']
            loader_tet = dset_loaders[str(names[args.t]) + '_test']
            loader_ss = dset_loaders[str(names[args.s]) + '_te']
            netF, netB, netC = train_target(loader_tgt, loader_tet_aug, loader_tet, loader_ss, netF, netB, netC, args)
            for j in range(len(names)):
                args.s = j
                if args.s == sss:
                    continue
                acc, _ = cal_acc(dset_loaders[str(names[args.s]) + '_test'], netF, netB, netC, False)
                log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(names[args.t][0].upper(), names[args.s][0].upper(), acc)
                args.out_file.write(log_str)
                print(log_str)
            args.s = sss
