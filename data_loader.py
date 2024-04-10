import os
import sys
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms


class ImageNetDataset(Dataset):
    def __init__(self,
                 args,
                 image_dir='C:/Users/14736/work_study/dataset/imagenet/archive/',
                 label2index_file='C:/Users/14736/work_study/dataset/imagenet/archive/Labels.json',
                 split='train'):
        super(ImageNetDataset).__init__()
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        self.image_list = []

        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)
        self.cat_list = sorted(os.listdir(self.image_dir))[:args.num_cat]

        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))[:self.args.num_perclass]
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]
        # label = self.cat_list.index(label)
        index = self.label2index[label]
        assert int(index) < self.args.num_cat
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        prefix = image_path.split('/')[-1].split('.')[0]
        return image, index, prefix

class TinyImageNetDataset(Dataset):
    def __init__(self, args, data_dir, split='train'):
        super(TinyImageNetDataset, self).__init__()
        self.args = args
        self.split = split
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        with open(os.path.join(data_dir, 'wnids.txt'), 'r') as f:
            self.wnids = [x.strip() for x in f.readlines()]
        self.label2index = {wnid: i for i, wnid in enumerate(self.wnids)}

        self.wnids = self.wnids[:args.num_cat]
        self.image_list = []
        self.labels = []

        if split == 'val':
            self.load_val_images_and_labels()
        else:
            self.load_images_and_labels()

    def load_val_images_and_labels(self):
        val_annotations_path = 'C:/Users/dataset/imagenet/tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt'
        class_image_count = {wnid: 0 for wnid in self.wnids}

        with open(val_annotations_path, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                img_file = tokens[0]
                wnid = tokens[1]
                if wnid in class_image_count and class_image_count[wnid] < self.args.num_perclass:
                    image_path = os.path.join(self.data_dir, 'images', img_file)
                    self.image_list.append(image_path)
                    self.labels.append(self.label2index[wnid])
                    class_image_count[wnid] += 1

    def load_images_and_labels(self):
        for wnid in self.wnids:
            wnid_path = os.path.join(self.data_dir, wnid, 'images')
            img_names = os.listdir(wnid_path)
            img_names = img_names[:self.args.num_perclass]
            for img_name in img_names:
                image_path = os.path.join(wnid_path, img_name)
                self.image_list.append(image_path)
                self.labels.append(self.label2index[wnid])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        prefix = os.path.splitext(os.path.basename(image_path))[0]
        return image, label, prefix

class CIFAR10Dataset(Dataset):
    def __init__(self,
                 args,
                 image_dir='C:/Users/14736/work_study/dataset/cifar10_png/cifar10/',
                 # image_dir='C:/Users/14736/work_study/new_dataset_for_test_2/cifar10/resnet18/origin_cw_fgsm_bim/',
                 split='train'):
        super(CIFAR10Dataset).__init__()
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ])
        self.image_list = []
        self.cat_list = sorted(os.listdir(self.image_dir))[:self.args.num_cat]
        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))[:self.args.num_perclass]
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]
        label = self.cat_list.index(label)
        label = torch.LongTensor([label]).squeeze()
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        prefix = image_path.split('/')[-1].split('.')[0]
        return image, label, prefix

class MNISTDataset(Dataset):
    def __init__(self,
                 args,
                 image_dir='C:/Users/14736/work_study/dataset/Mnist_png/mnist_png/',
                 split='train'):
        super(MNISTDataset).__init__()
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307, ), (0.3081, )),
                ])
        self.image_list = []
        self.cat_list = sorted(os.listdir(self.image_dir))[:self.args.num_cat]
        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))[:self.args.num_perclass]
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]
        label = self.cat_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path)
        image = self.transform(image)

        prefix = image_path.split('/')[-1].split('.')[0]
        return image, label, prefix


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()

    def init_param(self):
        self.gpus = torch.cuda.device_count()

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=1,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle
                        )
        return data_loader

if __name__ == '__main__':
    pass