import torch
from torch.nn import DataParallel
from datasets.train_loader import TrainLoader
from models import PulseDectector
from torch import optim
import argparse
from utils import create_logger, config_from_py, create_folder
import os


def train(model, data, train_configs, save_path, model_name, logger, device=torch.device('cpu')):
    model.train()
    max_epoch = train_configs["epoch"]
    init_lr = train_configs["learning_rate"]
    lr_step = train_configs['lr_step']
    save_step = train_configs["save_step"]
    verbose = train_configs["verbose"]
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    iter_per_epoch = len(data)
    for e in range(max_epoch):
        data_iter = iter(data)
        lr = init_lr * (0.1 ** (e // lr_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for it in range(iter_per_epoch):
            imgs, anns, gt_points = next(data_iter)
            imgs = imgs.to(device)
            anns = [ann.to(device) for ann in anns]
            gt_points = [gt_point.to(device) for gt_point in gt_points]
            loss = model(imgs, anns, gt_points=gt_points)
            loss_cls = loss['loss_cls'].mean()
            loss_reg = loss['loss_reg'] * 0.1
            loss_sum = loss_cls + loss_reg
            if (it+1) % verbose == 0:
                logger.info("Epoch:[{}] iter:[{}/{}] lr:{} loss_cls:{:.3f}  loss_reg:{:.4f} loss:{:.3f}".format(e, it, iter_per_epoch, lr,
                                                                        loss_cls, loss_reg, loss_sum.data))
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
        if (e+1)%save_step == 0:
            torch.save(
                {'epoch': e + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/{}_{}.pth".format(save_path, model_name, e+1))
            logger.info("save model to {}/{}_{}.pth".format(save_path, model_name, e+1))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a pulse detector")
    parser.add_argument('config', help='config file')
    parser.add_argument('--name', help='model name', default='resnet-fcos')
    parser.add_argument('--gpu', action='store_true',help='whether to use gpu', default=torch.cuda.is_available())
    parser.add_argument('--output', help='output path to save model', default='./cpks')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output):
        create_folder(args.output)
    # create a log file to save output information
    logger = create_logger("Train", args.output)
    logger.info('Train model：{}'.format(args.name))
    if args.gpu:
        assert torch.cuda.is_available()
        num_gpus = 1 # only single gpu training are supported TODO: distibute training
        logger.info('Runing on {} gpus'.format(num_gpus))
        device = torch.device('cuda')
    else:
        logger.info('Runing on cpu')
        num_gpus = 0
        device = torch.device('cpu')
    logger.info('Load configs from {}'.format(args.config))
    cfg = config_from_py(args.config)

    logger.info('Create model...')
    model = PulseDectector(cfg)
    model = model.to(device)
    # if num_gpus > 1:
    #     model = DataParallel(model)
    data = TrainLoader(cfg.data_set)

    train(model, data, cfg.train_configs, args.output, args.name, logger, device)