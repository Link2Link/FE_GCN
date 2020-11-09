import __init__
import numpy as np
import torch
from torch_geometric.data import DenseDataLoader
# import architecture
import utils.panda_dataset as dataset
import logging
from tqdm import tqdm
import os, sys
# import config
from DetecterData import *
import config
import vispy
from vispy.scene import visuals, SceneCanvas
from utils import color, visualize, graph, tools
from utils.tools import *

from sem_pretrain.code import architecture as semseg
from sem_pretrain.code import config
from gcn_lib.dense import BasicConv, GraphConv2d, PlainDynBlock2d, ResDynBlock2d, ResBlock2d, MLP
from torch.nn import Sequential as Seq
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
import torch.nn as nn
from torch.nn import Sequential as Seq
from itertools import chain
from utils.metrics import AverageMeter
import architecture
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
import utils.panda_dataset as dataset

def train(nomalizer, detector, train_loader, optimizer, scheduler, criterion, opt):
    opt.losses.reset()
    nomalizer.train()
    detector.train()
    with tqdm(train_loader) as tqdm_loader:
        for i, index in enumerate(tqdm_loader):
            opt.iter += 1
            desc = 'Epoch:{}  Iter:{}  [{}/{}]  Loss:{Losses.avg: .4f}' \
                .format(opt.epoch, opt.iter, i + 1, len(train_loader), Losses=opt.losses)
            tqdm_loader.set_description(desc)

            data_list = train_loader.dataset.data[index]
            obj_points_list, label, boxes = DatasetConcat.UnpackData(data_list, opt.device)

            optimizer.zero_grad()
            # nomalize
            features = torch.empty([0, nomallized_feature]).to(opt.device)
            positions = torch.empty([0, 3]).to(opt.device)
            for vertex in obj_points_list:
                position, feature = nomalizer(vertex)
                features = torch.cat([features, feature], dim=0)
                positions = torch.cat([positions, position], dim=0)

            # classify
            outputs = detector(features)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            opt.losses.update(loss.item())

            nomalizer_cpu = {k: v.cpu() for k, v in nomalizer.state_dict().items()}
            detector_cpu = {k: v.cpu() for k, v in detector.state_dict().items()}
            save_checkpoint({
                'epoch': opt.epoch,
                'state_dict': nomalizer_cpu,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, False, opt.ckpt_dir, 'nomalizer')

            save_checkpoint({
                'epoch': opt.epoch,
                'state_dict': detector_cpu,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, False, opt.ckpt_dir, 'detector')

global target_np
global pred_np
def test(nomalizer, detector, test_loader, opt):
    global target_np
    global pred_np
    nomalizer.eval()
    detector.eval()

    entire_num = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    with torch.no_grad():
        for i, index in enumerate(tqdm(test_loader)):
            data_list = train_loader.dataset.data[index]
            obj_points_list, label, boxes = DatasetConcat.UnpackData(data_list, opt.device)

            features = torch.empty([0, nomallized_feature]).to(opt.device)
            positions = torch.empty([0, 3]).to(opt.device)
            for vertex in obj_points_list:
                position, feature = nomalizer(vertex)
                features = torch.cat([features, feature], dim=0)
                positions = torch.cat([positions, position], dim=0)


            # classify
            outputs = detector(features)

            pred = outputs.max(dim=1)[1]
            pred_np = pred.cpu().numpy()
            target_np = label.cpu().numpy()

            entire_num += len(pred_np)
            TP += np.sum(pred_np[np.where(target_np==1)]==1)
            FP += np.sum(pred_np[np.where(target_np==0)]==1)
            FN += np.sum(pred_np[np.where(target_np==1)]==0)
            TN += np.sum(pred_np[np.where(target_np==0)]==0)

        Acc = (TP+TN) / (TP+TN+FN+FP)
        Precision = TP / (TP+FP)
        Recall = TP / (TP+FN)
        F1_score = 2*TP / (2*TP + FP + FN)
        opt.test_value = F1_score
        logging.info('TEST Epoch: [{}]\t Acc: {:.4f}\t Precision: {:.4f}\t Recall: {:.4f}\t F1_score: {:.4f}\t'.format(opt.epoch, Acc, Precision, Recall, F1_score))



if __name__ == '__main__':
    opt = config.OptInit().get_args()
    logging.info('===> Creating dataloader ...')
    trainset_list = [DetectionDataset(seq_num=i) for i in range(40)]
    trainset = DatasetConcat(trainset_list)
    train_loader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    testset_list = [DetectionDataset(seq_num=i) for i in range(40, 44)]
    testset = DatasetConcat(testset_list)
    test_loader = DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    logging.info('===> Loading the network ...')
    nomallized_feature = 256
    nomalizer = architecture.Nomalizer(in_channels=3, out_channels=nomallized_feature).to(opt.device)
    detector = architecture.ObjectDetection(channels=[nomallized_feature, 256, 64, 2]).to(opt.device)

    logging.info('===> nomalizer:')
    logging.info(nomalizer)

    logging.info('===> detector:')
    logging.info(detector)

    logging.info('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(chain(nomalizer.parameters(), detector.parameters()), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)

    opt.losses = AverageMeter()
    opt.test_value = 0.

    logging.info('===> start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        logging.info('Epoch:{}'.format(opt.epoch))
        train(nomalizer, detector, train_loader, optimizer, scheduler, criterion, opt)
        if opt.epoch % opt.eval_freq == 0 and opt.eval_freq != -1:
            test(nomalizer, detector, test_loader, opt)
        scheduler.step()



