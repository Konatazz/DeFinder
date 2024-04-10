import os
import copy
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from get_coverage import data_loader
from get_coverage import tool
from get_coverage import coverage
from get_coverage import constants

def evaluate_model_coverage(dataset, model_name, criterion_name, hyper,
                             image_size=32, batch_size=1, num_workers=0, num_class=10, num_per_class=20):
    """
    """
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if dataset == 'ImageNet':
        model = torchvision.models.__dict__[model_name](pretrained=False)
    elif dataset == 'CIFAR10':
        model = getattr(models, model_name)(pretrained=False)

    path = 'C:/Users/models/for_cifar10/state_dicts/state_dicts/resnet18.pt'
    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    model.eval()

    args = type('', (), {})()
    args.dataset = dataset
    args.image_size = image_size
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.num_class = num_class
    args.num_per_class = num_per_class
    TOTAL_CLASS_NUM, train_loader, test_loader, _ = data_loader.get_loader(args)

    input_size = (1, 3, image_size, image_size)
    random_data = torch.randn(input_size).to(DEVICE)
    layer_size_dict = tool.get_layer_output_sizes(model, random_data)
    criterion = getattr(coverage, criterion_name)(model, layer_size_dict, hyper=hyper)
    criterion.build(train_loader)
    criterion1 = copy.deepcopy(criterion)
    criterion1.assess(test_loader)

    return criterion1.current