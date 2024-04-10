import os
import cv2
import argparse
import copy
import utils
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import DeepLift
import cifar10_models
import imagenet_models
import mnist_models
import torchattacks
import imgaug.augmenters as iaa
from data_loader import *
from genetic_algorithm import *
from get_coverage import *


def random_pick(arr):
    idx = np.random.randint(0, len(arr))
    return arr[idx]


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['ImageNet', 'CIFAR10', 'MNIST'])
parser.add_argument('--model', type=str, default='vgg16_bn', choices=['resnet18', 'resnet34', 'vgg13_bn', 'vgg16_bn', 'mobilenet_v2', 'densenet121', 'densenet161', 'googlenet'])
parser.add_argument('--op', type=str, default='all', choices=['all', 'G', 'P', 'S', 'A', 'W'])
parser.add_argument('--num_cat', type=int, default=10, help='Number of categories to use')
parser.add_argument('--num_perclass', type=int, default=20, help='Number of images per class to use')
parser.add_argument('--output_root', type=str, default='')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()


# model = getattr(imagenet_models, args.model)(pretrained=False)
model = getattr(cifar10_models, args.model)(pretrained=False)
# model = getattr(mnist_models, args.model)()

ckpt_path = 'C:/Users/models/for_cifar10/state_dicts/state_dicts/vgg16_bn.pt'
model.load_state_dict(torch.load(ckpt_path))

model_ex = copy.deepcopy(model)
dl = DeepLift(model_ex)

if args.dataset == 'ImageNet':
    train_dataset = ImageNetDataset(args,
                    split='train',
                )
    test_dataset = ImageNetDataset(args,
                    split='val'
                )
elif args.dataset == 'CIFAR10':
    train_dataset = CIFAR10Dataset(args,
                    split='train',
                )
    test_dataset = CIFAR10Dataset(args,
                    split='test'
                )
elif args.dataset == 'MNIST':
    train_dataset = MNISTDataset(args,
                    split='train',
                )
    test_dataset = MNISTDataset(args,
                    split='test'
                )

data_loader_instance = DataLoader(args)
train_loader = data_loader_instance.get_loader(train_dataset, False)
test_loader = data_loader_instance.get_loader(test_dataset, False)

fgsm = torchattacks.FGSM(model, eps=0.2)
bim = torchattacks.BIM(model, eps=0.1, alpha=0.1)
pgd = torchattacks.PGD(model, eps=0.1, alpha=0.1)
cw = torchattacks.CW(model)
A = [fgsm, bim, pgd, cw]

correct_predictions = 0
total_predictions = 0

if __name__ == '__main__':
    dl = DeepLift(model_ex)
    original_openings = []
    attacked_fgsm_openings = []
    attacked_bim_openings = []
    attacked_pgd_openings = []
    attacked_cw_openings = []

    for i, (image, label, prefix) in enumerate(tqdm(test_loader)):
        image.requires_grad = True
        baselines = torch.zeros_like(image)
        logit = model(image)
        _, pred = logit.max(-1)
        attr = dl.attribute(image, target=pred, baselines=baselines)
        th, closing, opening = utils.attr2concept(attr[0])

        original_openings.append(opening)

        original_image_np = image[0].cpu().detach().numpy().transpose(1, 2, 0)
        original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min())
        opening_mask = torch.from_numpy(opening).type(torch.bool)
        opening_mask = opening_mask.unsqueeze(0).unsqueeze(0)
        opening_mask = opening_mask.expand(-1, 3, -1, -1)
        opening_only_image = image.clone()
        opening_only_image[~opening_mask] = 0
        opening_only_image_np = opening_only_image.squeeze().cpu().detach().numpy()
        opening_only_image_np = opening_only_image_np.transpose(1, 2, 0)
        opening_only_image_np = (opening_only_image_np - opening_only_image_np.min()) / (opening_only_image_np.max() - opening_only_image_np.min())

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        axs[0].imshow(original_image_np)
        axs[0].axis('off')
        axs[1].imshow(th, cmap='hot')
        axs[1].axis('off')
        axs[2].imshow(closing, cmap='hot')
        axs[2].axis('off')
        axs[3].imshow(opening, cmap='hot')
        axs[3].axis('off')
        axs[4].imshow(opening_only_image_np)
        axs[4].axis('off')
        plt.savefig(f"C:/Users/decision_maps_2/cifar10/vgg16_bn/origin_cw_fgsm_bim/origin/Decision_map_{i}.png")
        plt.close(fig)

    for i, (image, label, prefix) in enumerate(tqdm(test_loader)):
        adversarial_image = cw(image, label)

        adversarial_image.requires_grad = True
        baselines = torch.zeros_like(adversarial_image)
        logit = model(adversarial_image)
        _, pred = logit.max(-1)
        attr = dl.attribute(adversarial_image, target=pred, baselines=baselines)
        th, closing, opening = utils.attr2concept(attr[0])

        attacked_fgsm_openings.append(opening)

        original_image_np = image[0].cpu().detach().numpy().transpose(1, 2, 0)
        original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min())
        adversarial_image_np = adversarial_image[0].cpu().detach().numpy().transpose(1, 2, 0)
        adversarial_image_np = (adversarial_image_np - adversarial_image_np.min()) / (adversarial_image_np.max() - adversarial_image_np.min())

        opening_mask = torch.from_numpy(opening).type(torch.bool)
        opening_mask = opening_mask.unsqueeze(0).unsqueeze(0)
        opening_mask = opening_mask.expand(-1, 3, -1, -1)

        opening_only_image = adversarial_image.clone()
        opening_only_image[~opening_mask] = 0
        opening_only_image_np = opening_only_image.squeeze().cpu().detach().numpy()
        opening_only_image_np = opening_only_image_np.transpose(1, 2, 0)
        opening_only_image_np = (opening_only_image_np - opening_only_image_np.min()) / (opening_only_image_np.max() - opening_only_image_np.min())

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        axs[0].imshow(adversarial_image_np)
        axs[0].axis('off')
        axs[1].imshow(th, cmap='hot')
        axs[1].axis('off')
        axs[2].imshow(closing, cmap='hot')
        axs[2].axis('off')
        axs[3].imshow(opening, cmap='hot')
        axs[3].axis('off')
        axs[4].imshow(opening_only_image_np)
        axs[4].axis('off')
        plt.savefig(f"C:/Users/decision_maps_2/cifar10/vgg16_bn/origin_cw_fgsm_bim/cw/Decision_map_{i}.png")
        plt.close(fig)

    for i, (image, label, prefix) in enumerate(tqdm(test_loader)):
        adversarial_image = fgsm(image, label)

        adversarial_image.requires_grad = True
        baselines = torch.zeros_like(adversarial_image)
        logit = model(adversarial_image)
        _, pred = logit.max(-1)
        attr = dl.attribute(adversarial_image, target=pred, baselines=baselines)
        th, closing, opening = utils.attr2concept(attr[0])

        attacked_bim_openings.append(opening)

        original_image_np = image[0].cpu().detach().numpy().transpose(1, 2, 0)
        original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min())
        adversarial_image_np = adversarial_image[0].cpu().detach().numpy().transpose(1, 2, 0)
        adversarial_image_np = (adversarial_image_np - adversarial_image_np.min()) / (adversarial_image_np.max() - adversarial_image_np.min())

        opening_mask = torch.from_numpy(opening).type(torch.bool)
        opening_mask = opening_mask.unsqueeze(0).unsqueeze(0)
        opening_mask = opening_mask.expand(-1, 3, -1, -1)

        opening_only_image = adversarial_image.clone()
        opening_only_image[~opening_mask] = 0
        opening_only_image_np = opening_only_image.squeeze().cpu().detach().numpy()
        opening_only_image_np = opening_only_image_np.transpose(1, 2, 0)
        opening_only_image_np = (opening_only_image_np - opening_only_image_np.min()) / (opening_only_image_np.max() - opening_only_image_np.min())

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        axs[0].imshow(adversarial_image_np)
        axs[0].axis('off')
        axs[1].imshow(th, cmap='hot')
        axs[1].axis('off')
        axs[2].imshow(closing, cmap='hot')
        axs[2].axis('off')
        axs[3].imshow(opening, cmap='hot')
        axs[3].axis('off')
        axs[4].imshow(opening_only_image_np)
        axs[4].axis('off')
        plt.savefig(f"C:/Users/decision_maps_2/cifar10/vgg16_bn/origin_cw_fgsm_bim/fgsm/Decision_map_{i}.png")
        plt.close(fig)

    for i, (image, label, prefix) in enumerate(tqdm(test_loader)):
        adversarial_image = bim(image, label)

        adversarial_image.requires_grad = True
        baselines = torch.zeros_like(adversarial_image)
        logit = model(adversarial_image)
        _, pred = logit.max(-1)
        attr = dl.attribute(adversarial_image, target=pred, baselines=baselines)
        th, closing, opening = utils.attr2concept(attr[0])

        attacked_pgd_openings.append(opening)

        original_image_np = image[0].cpu().detach().numpy().transpose(1, 2, 0)
        original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min())
        adversarial_image_np = adversarial_image[0].cpu().detach().numpy().transpose(1, 2, 0)
        adversarial_image_np = (adversarial_image_np - adversarial_image_np.min()) / (adversarial_image_np.max() - adversarial_image_np.min())

        opening_mask = torch.from_numpy(opening).type(torch.bool)
        opening_mask = opening_mask.unsqueeze(0).unsqueeze(0)
        opening_mask = opening_mask.expand(-1, 3, -1, -1)

        opening_only_image = adversarial_image.clone()
        opening_only_image[~opening_mask] = 0
        opening_only_image_np = opening_only_image.squeeze().cpu().detach().numpy()
        opening_only_image_np = opening_only_image_np.transpose(1, 2, 0)
        opening_only_image_np = (opening_only_image_np - opening_only_image_np.min()) / (opening_only_image_np.max() - opening_only_image_np.min())

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        axs[0].imshow(adversarial_image_np)
        axs[0].axis('off')
        axs[1].imshow(th, cmap='hot')
        axs[1].axis('off')
        axs[2].imshow(closing, cmap='hot')
        axs[2].axis('off')
        axs[3].imshow(opening, cmap='hot')
        axs[3].axis('off')
        axs[4].imshow(opening_only_image_np)
        axs[4].axis('off')
        plt.savefig(f"C:/Users/decision_maps_2/cifar10/vgg16_bn/origin_cw_fgsm_bim/bim/Decision_map_{i}.png")
        plt.close(fig)

    # for i, (image, label, prefix) in enumerate(tqdm(test_loader)):
    #     # 选择攻击方式（fgsm, bim, pgd, cw）
    #     adversarial_image = cw(image, label)
    #
    #     adversarial_image.requires_grad = True
    #     baselines = torch.zeros_like(adversarial_image)
    #     logit = model(adversarial_image)
    #     _, pred = logit.max(-1)
    #     attr = dl.attribute(adversarial_image, target=pred, baselines=baselines)
    #     th, closing, opening = utils.attr2concept(attr[0])
    #
    #     attacked_cw_openings.append(opening)
    #
    #     original_image_np = image[0].cpu().detach().numpy().transpose(1, 2, 0)
    #     original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min())
    #     adversarial_image_np = adversarial_image[0].cpu().detach().numpy().transpose(1, 2, 0)
    #     adversarial_image_np = (adversarial_image_np - adversarial_image_np.min()) / (adversarial_image_np.max() - adversarial_image_np.min())
    #
    #     opening_mask = torch.from_numpy(opening).type(torch.bool)
    #     opening_mask = opening_mask.unsqueeze(0).unsqueeze(0)
    #     opening_mask = opening_mask.expand(-1, 3, -1, -1)
    #
    #     opening_only_image = adversarial_image.clone()
    #     opening_only_image[~opening_mask] = 0
    #     opening_only_image_np = opening_only_image.squeeze().cpu().detach().numpy()
    #     opening_only_image_np = opening_only_image_np.transpose(1, 2, 0)
    #     opening_only_image_np = (opening_only_image_np - opening_only_image_np.min()) / (opening_only_image_np.max() - opening_only_image_np.min())
    #
    #     fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    #     axs[0].imshow(adversarial_image_np)
    #     axs[0].axis('off')
    #     axs[1].imshow(th, cmap='hot')
    #     axs[1].axis('off')
    #     axs[2].imshow(closing, cmap='hot')
    #     axs[2].axis('off')
    #     axs[3].imshow(opening, cmap='hot')
    #     axs[3].axis('off')
    #     axs[4].imshow(opening_only_image_np)
    #     axs[4].axis('off')
    #     plt.savefig(f"C:/Users/decision_maps_2/cifar10/densenet121/origin_fgsm_bim_pgd_cw/cw/Decision_map_{i}.png")
    #     plt.close(fig)

    for i in tqdm(range(len(original_openings)), desc='Processing images'):
        # intersection = np.logical_and(original_openings[i], attacked_fgsm_openings[i])
        intersection_1 = np.logical_and(original_openings[i], attacked_fgsm_openings[i])
        intersection_2 = np.logical_and(intersection_1, attacked_bim_openings[i])
        # intersection_3 = np.logical_and(intersection_2, attacked_pgd_openings[i])
        intersection = np.logical_and(intersection_2, attacked_pgd_openings[i])

        original_image = test_dataset[i][0].numpy().transpose(1, 2, 0)
        original_image = original_image.astype(float)
        mask = np.zeros_like(original_image)
        mask[intersection] = 1
        highlighted_image = original_image * mask
        highlighted_image = (highlighted_image - highlighted_image.min()) / (highlighted_image.max() - highlighted_image.min() + 1e-8)
        plt.imsave(f"C:/Users/decision_maps_2/cifar10/vgg16_bn/origin_cw_fgsm_bim/intersections/intersections_image_{i}.png", highlighted_image)

        attacked_image = genetic_algorithm(original_image, mask, model, pop_size=70, num_generations=20)
        attacked_image = (attacked_image - attacked_image.min()) / (attacked_image.max() - attacked_image.min())
        plt.imsave(f"C:/Users/decision_maps_2/cifar10/vgg16_bn/origin_cw_fgsm_bim/attacked_images/attacked_image_{i}.png", attacked_image)

        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        test_loader.dataset.classes = classes
        label = test_dataset[i][1]
        class_name = test_loader.dataset.classes[label]
        save_dir = f"C:/Users/new_dataset_for_test_2/cifar10/vgg16_bn/origin_cw_fgsm_bim/test/{class_name}"
        os.makedirs(save_dir, exist_ok=True)
        new_cifar10_path = f"{save_dir}/attacked_image_{i}.png"
        plt.imsave(new_cifar10_path, attacked_image)

    # correct_predictions = 0
    # total_predictions = 0
    for i in tqdm(range(len(original_openings)), desc='Processing images'):
        attacked_image = plt.imread(f"C:/Users/decision_maps_2/cifar10/vgg16_bn/origin_cw_fgsm_bim/attacked_images/attacked_image_{i}.png")

        if attacked_image.shape[2] == 4:
            attacked_image = attacked_image[:, :, :3]
        attacked_image_tensor = transforms.ToTensor()(attacked_image).unsqueeze(0).float().to('cuda:0')

        model.eval()
        with torch.no_grad():
            preds = model(attacked_image_tensor)
            predicted_class = preds.argmax(dim=1)

        actual_label = test_dataset[i][1]
        label = test_dataset[i][1]
        total_predictions += 1
        if predicted_class.item() == actual_label:
            correct_predictions += 1

        plt.imshow(attacked_image)
        plt.title(f"Image {i} - Predicted class: {predicted_class.item()}")
        plt.savefig(f"C:/Users/decision_maps_2/cifar10/vgg16_bn/origin_cw_fgsm_bim/prediction_results/attacked_image_{i}_prediction.png")

    failure_rate = (total_predictions - correct_predictions) / total_predictions
    print(f"Overall Failure Rate: {failure_rate:.2f}")
