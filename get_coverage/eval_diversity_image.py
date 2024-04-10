import os
import copy
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import data_loader
import utility
import models
import tool
import coverage
import constants

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['ImageNet', 'CIFAR10'])
parser.add_argument('--model', type=str, default='googlenet', choices=['resnet18', 'resnet34', 'vgg13_bn', 'vgg16_bn', 'mobilenet_v2', 'densenet121', 'densenet161', 'googlenet'])
parser.add_argument('--criterion', type=str, default='NBC',
                    choices=['NLC', 'NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC',
                    'LSC', 'DSC', 'MDSC'])
parser.add_argument('--output_dir', type=str, default='./test_folder')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_class', type=float, default=10)
parser.add_argument('--num_per_class', type=float, default=20)
parser.add_argument('--hyper', type=float, default=None)
args = parser.parse_args()
args.exp_name = ('%s-%s-%s-%s' % (args.dataset, args.model, args.criterion, args.hyper))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

utility.make_path(args.output_dir)
fp = open('%s/%s.txt' % (args.output_dir, args.exp_name), 'w')

utility.log(('Dataset: %s \nModel: %s \nCriterion: %s \nHyper-parameter: %s' % (
    args.dataset, args.model, args.criterion, args.hyper
)), fp)

USE_SC = args.criterion in ['LSC', 'DSC', 'MDSC']

if args.dataset == 'ImageNet':
    model = torchvision.models.__dict__[args.model](pretrained=False)
    path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
    assert args.image_size == 128
    assert args.num_class <= 1000
elif args.dataset == 'CIFAR10':
    model = getattr(models, args.model)(pretrained=False)
    path = 'C:/Users/14736/work_study/models/for_cifar10/state_dicts/state_dicts/googlenet.pt'
    assert args.image_size == 32
    assert args.num_class <= 10

TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)

model.load_state_dict(torch.load(path))
model.to(DEVICE)
model.eval()

if __name__ == '__main__':

    input_size = (1, args.nc, args.image_size, args.image_size)
    random_data = torch.randn(input_size).to(DEVICE)
    layer_size_dict = tool.get_layer_output_sizes(model, random_data)

    num_neuron = 0
    for layer_name in layer_size_dict.keys():
        num_neuron += layer_size_dict[layer_name][0]
    print('Total %d layers: ' % len(layer_size_dict.keys()))
    print('Total %d neurons: ' % num_neuron)

    if USE_SC:
        criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=args.hyper, min_var=1e-5, num_class=TOTAL_CLASS_NUM)
    else:
        criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=args.hyper)

    criterion.build(train_loader)
    if args.criterion not in ['CC', 'TKNP', 'LSC', 'DSC', 'MDSC']:
        criterion.assess(train_loader)
    '''
    For LSC/DSC/MDSC/CC/TKNP, initialization with training data is too slow (sometimes may
    exceed the memory limit). You can skip this step to speed up the experiment, which
    will not affect the conclusion because we only compare the relative order of coverage
    values, rather than the exact numbers.
    '''
    utility.log('Initial coverage: %d' % criterion.current, fp)  # 记录初始覆盖率

    criterion1 = copy.deepcopy(criterion)
    criterion1.assess(test_loader)    # 评估测试数据集的覆盖情况
    utility.log(('Test: %f, increase: %f' % (criterion1.current, criterion1.current - criterion.current)), fp)
    del criterion1

    # times = 1
    # for times in [1, 10]:
    #     criterion2 = copy.deepcopy(criterion)   # 再次创建覆盖标准的深拷贝，用于评估增强数据集
    #     for i, (old_image, label) in enumerate(seed_loader):
    #         for j in tqdm(range(times * len(list(seed_loader)))):
    #             image = old_image + torch.clamp(torch.randn(old_image.size()), -0.2, 0.2)  # 对原始图像应用随机噪声
    #             if USE_SC:
    #                 criterion2.step(image.to(DEVICE), label.to(DEVICE))   # 更新覆盖标准状态
    #             else:
    #                 criterion2.step(image.to(DEVICE))
    #         break
    #     utility.log(('%s x%d: %f, increase: %f' % (args.dataset, times, criterion2.current, criterion2.current - criterion.current)), fp)
    #     del criterion2
