import __init__
import numpy as np
import torch
from torch_geometric.data import DenseDataLoader
import architecture
import utils.panda_dataset as dataset
import logging
from tqdm import tqdm
import os
import config
from torch.utils.data import ConcatDataset
from utils.metrics import AverageMeter
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
import gc

def train(model, train_loader, optimizer, scheduler, criterion, opt):
    opt.losses.reset()
    model.train()
    with tqdm(train_loader) as tqdm_loader:
        for i, data in enumerate(tqdm_loader):
            opt.iter += 1
            # tqdm progress bar
            desc = 'Epoch:{}  Iter:{}  [{}/{}]  Loss:{Losses.avg: .4f}' \
                .format(opt.epoch, opt.iter, i + 1, len(train_loader), Losses=opt.losses)
            tqdm_loader.set_description(desc)
            data = data.to(opt.device)
            inputs = torch.cat((data.pos.transpose(1, 2).unsqueeze(3), data.x.unsqueeze(1).unsqueeze(3)), 1)
            gt = data.y.to(opt.device).long()
            # ------------------ zero, output, loss
            optimizer.zero_grad()
            out = model(inputs, data.edge_index)
            loss = criterion(out, gt)

            # ------------------ optimization
            loss.backward()
            optimizer.step()
            opt.losses.update(loss.item())
        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        print(opt.ckpt_dir)
        save_checkpoint({
            'epoch': opt.epoch,
            'state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, False, opt.ckpt_dir, 'deepgcn')

def test(model, loader, opt):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))
    model.eval()
    pred_co = np.empty(shape=[0, 81920])
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            if not opt.multi_gpus:
                data = data.to(opt.device)
            data = data.to(opt.device)
            inputs = torch.cat((data.pos.transpose(1, 2).unsqueeze(3), data.x.unsqueeze(1).unsqueeze(3)), 1)
            gt = data.y.to(opt.device).long()
            out = model(inputs, data.edge_index)
            pred = out.max(dim=1)[1]
            pred_np = pred.cpu().numpy()
            target_np = gt.cpu().numpy()
            pred_co = np.append(pred_co, pred_np, axis=0)
            for cl in range(opt.n_classes):
                cur_gt_mask = (target_np == cl)
                cur_pred_mask = (pred_np == cl)
                I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                Is[i, cl] = I
                Us[i, cl] = U
    ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
    ious[np.isnan(ious)] = 1
    iou = np.mean(ious)

    opt.test_value = iou
    logging.info('TEST Epoch: [{}]\t mIoU: {:.4f}\t'.format(opt.epoch, opt.test_value))
    for cl in range(opt.n_classes):
        logging.info("===> mIOU for class {}: {}".format(cl, ious[cl]))
    file = os.path.join(opt.pred_dir, "pred_on_epoch_{epo}.npy".format(epo=opt.epoch))
    np.save(file, pred_co)
    logging.info('===> Saving pred to {file} ...'.format(file=file))

if __name__ == '__main__':
    opt = config.OptInit().get_args()
    logging.info('===> Creating dataloader ...')
    seq_num = dataset.return_seq(opt.data_dir)

    seq_test = seq_num[-4:]
    logging.info('===> Test on ' + '_'.join(seq_test))
    test_dataset_list = [dataset.PANDASET(root=opt.data_dir,seq_num=seq) for seq in seq_test]
    test_dataset = ConcatDataset(test_dataset_list)
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    opt.n_classes = 5

    logging.info('===> Loading the network ...')
    model = architecture.DenseDeepGCN(opt).to(opt.device)
    # model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)
    checkpoint = torch.load("/home/llx/code/GCN_Detecter/example/semseg/pretrain/20201104-001819/deepgcn_ckpt_30_best.pth")
    ckpt_model_state_dict = checkpoint['state_dict']
    # model_dict = model.state_dict()
    model.load_state_dict(ckpt_model_state_dict)


    logging.info(model)

    logging.info('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    # optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    logging.info('===> Init Metric ...')
    opt.losses = AverageMeter()
    opt.test_value = 0.

    logging.info('===> start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        logging.info('Epoch:{}'.format(opt.epoch))

        seq_train = seq_num[0:10]
        logging.info('===> Train on ' + '_'.join(seq_train))
        train_dataset_list = [dataset.PANDASET(root=opt.data_dir, seq_num=seq) for seq in seq_train]
        train_dataset = ConcatDataset(train_dataset_list)
        train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
        train(model, train_loader, optimizer, scheduler, criterion, opt)
        del train_dataset_list, train_dataset, train_loader
        gc.collect()

        seq_train = seq_num[10:20]
        logging.info('===> Train on ' + '_'.join(seq_train))
        train_dataset_list = [dataset.PANDASET(root=opt.data_dir, seq_num=seq) for seq in seq_train]
        train_dataset = ConcatDataset(train_dataset_list)
        train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
        train(model, train_loader, optimizer, scheduler, criterion, opt)
        del train_dataset_list, train_dataset, train_loader
        gc.collect()

        seq_train = seq_num[20:30]
        logging.info('===> Train on ' + '_'.join(seq_train))
        train_dataset_list = [dataset.PANDASET(root=opt.data_dir, seq_num=seq) for seq in seq_train]
        train_dataset = ConcatDataset(train_dataset_list)
        train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
        train(model, train_loader, optimizer, scheduler, criterion, opt)
        del train_dataset_list, train_dataset, train_loader
        gc.collect()

        seq_train = seq_num[30:40]
        logging.info('===> Train on ' + '_'.join(seq_train))
        train_dataset_list = [dataset.PANDASET(root=opt.data_dir, seq_num=seq) for seq in seq_train]
        train_dataset = ConcatDataset(train_dataset_list)
        train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
        train(model, train_loader, optimizer, scheduler, criterion, opt)
        del train_dataset_list, train_dataset, train_loader
        gc.collect()

        if opt.epoch % opt.eval_freq == 0 and opt.eval_freq != -1:
            test(model, test_loader, opt)
        scheduler.step()
    logging.info('Saving the final model.Finish!')
