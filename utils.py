import torch
import torch.nn as nn
import math
irange = range
import os
import random
import numpy as np
import shutil
import json
import cv2
import torchvision
import torchvision.transforms as transforms

def mask_ratio(src):
    src_mask = src == 255
    ones = np.ones(src.shape
        )
    return src_mask.astype(np.float32).sum() / ones.sum()

def IoU(src, tgt):
    src_mask = src == 255
    tgt_mask = tgt == 255
    return (src_mask & tgt_mask).astype(np.float32).sum() / (src_mask | tgt_mask).astype(np.float32).sum()


def segment_org(org_tensor, mask):
    # print('org_tensor: ', org_tensor.size())
    if isinstance(org_tensor, torch.Tensor):
        org_np = org_tensor.squeeze().cpu().detach().numpy()
    else:
        org_np = org_tensor
    for i in range(3):
        org_np[i] *= (mask == 255).astype(np.uint8)
    org_np = np.transpose(org_np, (1, 2, 0))
    return org_np * 255

def segment_org_white(org_tensor, mask):
    if isinstance(org_tensor, torch.Tensor):
        org_np = org_tensor.squeeze().cpu().detach().numpy()
    else:
        org_np = org_tensor
    for i in range(3):
        org_np[i] *= (mask == 255).astype(np.uint8)
        org_np[i] += (1 * (mask != 255)).astype(np.uint8)
    org_np = np.transpose(org_np, (1, 2, 0))
    return org_np * 255

def segment_org_green(org_tensor, mask):
    if isinstance(org_tensor, torch.Tensor):
        org_np = org_tensor.squeeze().cpu().detach().numpy()
    else:
        org_np = org_tensor
    for i in range(3):
        org_np[i] *= (mask == 255).astype(np.uint8)
    org_np[1] += (1 * (mask != 255)).astype(np.uint8)
    org_np = np.transpose(org_np, (1, 2, 0))
    return org_np * 255

def attr2concept(attr):
    if isinstance(attr, torch.Tensor):
        pixel0 = attr.squeeze().cpu().detach().numpy()
    else:
        pixel0 = attr

    index = np.where(pixel0 > 0)[0]
    pixel0_pos = pixel0[index]
    pixel0_pos = pixel0_pos.mean(0)
    pixel0_pos -= pixel0_pos.min()
    pixel0_pos /= pixel0_pos.max()
    pixel0_pos *= 255
    pixel0_pos = pixel0_pos.astype(np.uint8)
    _, th = cv2.threshold(pixel0_pos, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    kernel2 = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    return th, closing, opening

def attr2concept_mnist(attr):
    if isinstance(attr, torch.Tensor):
        pixel0 = attr.cpu().detach().numpy()
    else:
        pixel0 = attr
    index = np.where(pixel0 > 0)[0]
    pixel0_pos = pixel0[index]
    pixel0_pos = pixel0_pos.mean(0)
    pixel0_pos -= pixel0_pos.min()
    pixel0_pos /= pixel0_pos.max()
    pixel0_pos *= 255
    pixel0_pos = pixel0_pos.astype(np.uint8)
    _, th = cv2.threshold(pixel0_pos, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    kernel2 = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    return th, closing, closing


def image_normalize(image, dataset):
    if dataset == 'CIFAR10':
       transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif dataset == 'MNIST':
        transform = transforms.Normalize((0.1307, ), (0.3081, ))
    else:
        raise NotImplementedError
    return transform(image)

def image_normalize_inv(image, dataset):
    if dataset == 'CIFAR10':
        transform = NormalizeInverse((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = NormalizeInverse((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif dataset == 'MNIST':
        transform = NormalizeInverse((0.1307, ), (0.3081, ))
    else:
        raise NotImplementedError
    return transform(image)

def accuracy(scores, targets, k=1):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_params(json_file):
    with open(json_file) as f:
        return json.load(f)

def get_batch(data_loader):
    while True:
        for batch in data_loader:
            yield batch

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:
        if tensor.size(0) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()
        if range is not None:
            assert isinstance(range, tuple), "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr).convert('RGB')
    
    # (width, height) = im.size[:2]
    # im = im.resize((width*3, height*3), Image.ANTIALIAS)
    
    im.save(filename)

def save_image_resize(tensor, filename, ratio=1, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr).convert('RGB')
    
    (width, height) = im.size[:2]
    im = im.resize((width*ratio, height*ratio), Image.ANTIALIAS)
    
    im.save(filename)