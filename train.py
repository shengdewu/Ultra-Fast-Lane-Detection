import torch, os, datetime
import numpy as np

from model.model import parsingNet
from data.dataloader import get_train_loader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time

import configs.tusimple
import data.constant
import cv2

import platform

def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.cuda(), seg_label.cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out': seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img, cls_label.long()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'img': img}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0 and logger is not None:
            logger.add_scalar('loss/' + loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_aux):
    net.train()
    progress_bar = dist_tqdm(train_loader)
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        if global_step % 50 == 0:
            img = results['img'].numpy()
            cls_out = results['cls_out'].numpy()
            cls_label = results['cls_label'].numpy()
            bs, _, _, _ = img.shape
            for b in range(bs):
                predict = reconstruct(cls_out[b], img[b].copy())
                ground = reconstruct(cls_label[b], img[b].copy())
                all_img = np.hstack([ground, predict])
                cv2.imwrite(configs.tusimple.out_path + '/' + str(epoch) + '-' + str(b) + '.png', all_img)

        update_metrics(metric_dict, results)
        if logger is not None:
            if global_step % 20 == 0:
                for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                    logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar, 'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss='%.3f' % float(loss),
                                     data_time='%.3f' % float(t_data_1 - t_data_0),
                                     net_time='%.3f' % float(t_net_1 - t_net_0),
                                     **kwargs)
        t_data_0 = time.time()

def reconstruct(cls, img):
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    c, h, w = img.shape
    lane_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    row_anchors = data.constant.tusimple_row_anchor
    if h != 720:
        row_anchors = [int(i*720.0/h) for i in row_anchors]

    for j in range(cls.shape[1]):
        out_i = cls[:, j]
        to_pts = [int(round(loc * 1280 / configs.tusimple.griding_num)) if loc != configs.tusimple.griding_num else -2 for loc in out_i]

        points = [(w, h) for h, w in zip(row_anchors, to_pts)]
        for l in points:
            if l[0] == -2:
                continue
            cv2.circle(lane_img, l, radius=3, color=color[j], thickness=3)
    return lane_img

def pring_net_struct(path, net):
    with open(os.path.join(path, 'pytorch_net.txt'), 'w') as w:
        net_struct = '{}\n'.format(net)
        w.write(net_struct)
        w.write('-----------------param-------------------\n')
        for name, param in net.named_parameters():
            w.write('{} : {}\n'.format(name, param.size()))
        w.write('-----------------param-------------------\n')
        for param in net.state_dict():
            w.write('{}-{}\n'.format(param, net.state_dict()[param].size()))
    return

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    is_win = False
    if platform.system() == 'Windows':
      is_win = True
    args, cfg = merge_config(is_win)

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    train_loader, cls_num_per_lane = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_aux, distributed)

    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4), use_aux=cfg.use_aux)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    optimizer = get_optimizer(net, cfg)
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    #logger = get_logger(work_dir, cfg)
    #cp_projects(work_dir)

    pring_net_struct(cfg.log_path, net)

    for epoch in range(resume_epoch, cfg.epoch):
        train(net, train_loader, loss_dict, optimizer, scheduler, None, epoch, metric_dict, cfg.use_aux)

        save_model(net, optimizer, epoch, work_dir, distributed)
    #logger.close()
