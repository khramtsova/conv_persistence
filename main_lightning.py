from os import listdir
from os.path import isfile, join

import pytorch_lightning as pl
from argparse import ArgumentParser
import wandb
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from torchvision import transforms
import torchmetrics

import torch.nn.functional as F

from pytorch_lightning.metrics.classification import Accuracy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import functional as FM

from pytorch_lightning.loggers import CSVLogger
import torchvision.models as models

#from lib.utils.metrics import accuracy
from networks.wideresnet import WideResNet
from networks.wideresnet_instance_norm import WideResNet as WideResNetInstanceNorm
from networks.rand_conv import RandConvModule

from data.CIFAR_data_module import CIFARDataModule
from pytorch_lightning.loggers import WandbLogger
#from topo.topo_utils import get_diagrams_feature_vectors, wasserstein_d


class Net(pl.LightningModule):

    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.save_hyperparameters()

        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)

        if self.args.norm == "InstanceNorm2ds":
            self.net = WideResNetInstanceNorm(depth=args.depth, num_classes=10, widen_factor=args.width)
        else:
            self.net = WideResNet(depth=args.depth, num_classes=10, widen_factor=args.width)

        self.rand_module = RandConvModule(
            in_channels=3,
            out_channels=3,
            kernel_size=[1, 3, 5, 7],
            mixing=True,
            identity_prob=0.0,
            rand_bias=False,
            distribution='kaiming_normal',
            data_mean=data_mean,
            data_std=data_std,
            clamp_output=False,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.metric_name = 'acc'
        self.metric = torchmetrics.Accuracy() #accuracy

    def configure_optimizers(self):

        if self.args.SGD:
            print("Using SGD optimizer")
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.args.lr,
                                  momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay,
                                  nesterov=self.args.nesterov)
        else:
            print("Using Adam optimizer")
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

        if self.args.scheduler == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                  step_size=self.args.step_size,
                                                  gamma=self.args.gamma)
        elif self.args.scheduler == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                       milestones=self.args.milestones,
                                                       gamma=self.args.gamma)
        elif self.args.scheduler == 'CosLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             self.n_steps_per_epoch * self.args.n_epoch)
        elif not self.args.scheduler:
            return optimizer
        else:
            raise NotImplementedError()

        return [optimizer], [scheduler]

    def on_train_start(self):
        pl.seed_everything(self.args.rand_seed)

    def training_step(self, batch, batch_indx):
        inputs, targets = batch

        self.rand_module.randomize()
        inputs0 = self.rand_module(inputs)
        outputs = self.net(inputs0)
        loss = self.criterion(outputs, targets)
        metric = self.metric(outputs, targets)

        self.rand_module.randomize()
        outputs1 = self.net(self.rand_module(inputs))
        # loss += self.criterion(outputs1, targets)

        self.rand_module.randomize()
        outputs2 = self.net(self.rand_module(inputs))
        # loss += self.criterion(outputs2, targets)

        # noinspection DuplicatedCode
        p_clean, p_aug1, p_aug2 = F.softmax(outputs, dim=1), \
                                  F.softmax(outputs1, dim=1), \
                                  F.softmax(outputs2, dim=1)
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        inv_loss = ( F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                     F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                     F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        loss += inv_loss
        self.log("train/acc", metric)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_indx, dataset_idx):
        inputs, targets = batch
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        metric = self.metric(outputs, targets)
        if dataset_idx < len(self.args.corruptions):
            self.log("test_acc/", metric)
            self.log("test_loss/", loss)
        else:
            self.log("val/acc", metric)
            self.log("val/loss", loss)

    def test_step(self, batch, batch_indx, dataset_idx):
        inputs, targets = batch
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        metric = self.metric(outputs, targets)
        self.log("test_acc/"+self.args.corruptions[dataset_idx], metric)
        self.log("test_loss/"+self.args.corruptions[dataset_idx], loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--rand_seed', default=0, type=int)
        # model
        parser.add_argument('--model', default='metawrn')
        parser.add_argument('--depth', default=16, type=int)
        parser.add_argument('--width', default=4, type=int)
        parser.add_argument('--hidden_dim', default=256, type=int)
        parser.add_argument('--norm', default='InstanceNorm2d', choices=['BatchNorm2d', 'InstanceNorm2d', 'GroupNorm'],
                            type=str)
        # dataloader
        parser.add_argument('--dataset', default='cifar10', type=str)
        parser.add_argument('--num_classes', default=10, type=int)
        parser.add_argument('--num_aug', default=0, type=int)
        parser.add_argument('--img_size', default=32, type=int)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--corruptions', default=[
            'fog', 'snow', 'frost', 'zoom_blur',
            'defocus_blur', 'glass_blur', 'motion_blur', 'shot_noise',
            'impulse_noise', 'gaussian_noise', 'jpeg_compression', 'pixelate',
            'elastic_transform', 'brightness', 'contrast'
        ], type=list)
        # optimizer
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument('--num_epochs', default=181, type=int)
        # log
        parser.add_argument('--log_path', default='./log', type=str)
        parser.add_argument('--outputs', default='outputs', type=str)
        parser.add_argument('--checkpoints', default='checkpoints', type=str)
        # others
        # parser.add_argument('note', help='add a "_" at the beginning')
        # parser.add_argument('comment', help='descript the current run')
        parser.add_argument('--SGD', '-sgd', action='store_true', help='use optimizer')
        parser.add_argument('--nesterov', '-nest', action='store_true', help='use nesterov momentum')
        parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float, help='weight decay')
        parser.add_argument('--momentum', '-mmt', default=0.9, type=float, help='momentum')
        parser.add_argument('--scheduler', '-sch', type=str, default='',
                            help='type of lr scheduler, StepLR/MultiStepLR/CosLR')
        parser.add_argument('--step_size', '-stp', type=int, default=30, help='fixed step size for StepLR')
        parser.add_argument('--milestones', '-milestones', type=int, nargs='+', help='milestone for MultiStepLR')
        parser.add_argument('--gamma', '-gm', type=float, default=0.2, help='reduce rate for step scheduler')
        parser.add_argument('--power', '-power', default=0.9, help='power for poly scheduler')

        parser.add_argument('--only_severe_corruption', action='store_true', help='only consider level5 corruptions')

        parser.add_argument('--device', default='cuda:0')
        return parser

    @staticmethod
    def add_program_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--test_latest', '-tl', action='store_true',
                            help='test the last saved model instead of the best one')
        parser.add_argument('--test_target', '-tt', action='store_true',
                            help='test the best model on target domain')
        parser.add_argument('--data_path', default='/opt/data', type=str)
        parser.add_argument('--base_c_path', default='/opt/data/cifar_10_corruptions/', type=str)
        parser.add_argument('--data_dir', '-data_dir', type=str,  default="./data", help='directory with data')
        parser.add_argument('--log_dir', '-log_dir', type=str, default="./log", help='directory for logs')
        return parser


if __name__ == "__main__":
    # main()
    parser = ArgumentParser(description='CIFAR10 training')
    parser = Net.add_model_specific_args(parser)
    parser = Net.add_program_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print("HERE")

    data = CIFARDataModule(args)

    image_model = Net(args=args)

    checkpoint_callback = ModelCheckpoint(  # monitor="metric_val/acc",
        # period=10,
        # every_n_epochs=10,
        # mode="max"
        # save_top_k=-1,
        save_weights_only=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    #logger = TensorBoardLogger(args.log_dir,
    #                           name="del")

    logger = WandbLogger(save_dir=args.log_dir,
                         project='RandConv_CIFAR',
                         job_type='train',
                         entity='name')
    logger.watch(image_model)

    trainer = pl.Trainer.from_argparse_args(args,
                                            # checkpoint_callback=checkpoint_callback,
                                            callbacks=checkpoint_callback,
                                            # callbacks=[lr_monitor],
                                            # progress_bar_refresh_rate=300,
                                            logger=logger,
                                            # num_sanity_val_steps=-1,
                                            num_nodes=1,
                                            terminate_on_nan=True,
                                            )
    #trainer.test(image_model, data)
    trainer.fit(image_model, data)
