import json, time, random, os, GPUtil

def get_available_gpus(min_memory_available):
    """Returns a list of IDs for GPUs with more than min_memory_available MB of memory."""
    GPUs = GPUtil.getGPUs()
    available_gpus = [gpu.id for gpu in GPUs if gpu.memoryFree > min_memory_available]
    return available_gpus

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score
import torchvision.transforms as tt
from torch.utils.data import DataLoader
import albumentations as A
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import _LRScheduler

global mean_origin
global std_origin

mean_origin = [0.48145466, 0.4578275, 0.40821073]
std_origin = [0.26862954, 0.26130258, 0.27577711]

def fgsm_attack(model, input_, opt, criterion, num_classes, labels, epsilon):
    # alpha = epsilon / opt.p_itr
    inputs = input_.clone().detach()
    inputs.requires_grad = True

    features_logits, image_features, text_features = model(inputs)
    prompts_loss = torch.zeros(1).to(features_logits.device)


    # if i == 0:
    stored_image_features = image_features
    stored_text_features = text_features


    if len(features_logits.shape) == 2:
        temp_features_logits = features_logits
    elif len(features_logits.shape) == 3:
        temp_features_logits = features_logits[:, 0:1, :].squeeze(1)

    loss = get_loss(opt, criterion, temp_features_logits, num_classes, labels, prompts_loss)
    # optimizer.zero_grad()
    model.zero_grad()
    loss.backward()


    zero_indices = (labels == 0).nonzero().squeeze()
    one_indices = (labels == 1).nonzero().squeeze()
    
    # Generate adversarial examples
    perturbed_image_full = input_.clone().detach()

    if zero_indices.nelement() == 1:
        zero_indices.unsqueeze_(0)
    idx = len(zero_indices)
    # half_idx = int(idx/2)
    half_idx = int(idx*opt.divideNum)
    data_grad = inputs.grad.data[zero_indices[half_idx:idx]]
    perturbing_image = perturbed_image_full[zero_indices[half_idx:idx]].clone().detach()

    perturbing_image[:, 0, :, :] = perturbing_image[:, 0, :, :] * std_origin[0] + mean_origin[0]
    perturbing_image[:, 1, :, :] = perturbing_image[:, 1, :, :] * std_origin[1] + mean_origin[1]
    perturbing_image[:, 2, :, :] = perturbing_image[:, 2, :, :] * std_origin[2] + mean_origin[2]

    sign_data_grad = data_grad.sign()
    clamping_image = perturbing_image + epsilon*sign_data_grad
    # eta = torch.clamp(clamping_image - perturbing_image, min=-0.1, max=0.1)
    perturbed_image = torch.clamp(clamping_image, 0, 1)

    perturbed_image[:, 0, :, :] = (perturbed_image[:, 0, :, :] - mean_origin[0]) / std_origin[0]
    perturbed_image[:, 1, :, :] = (perturbed_image[:, 1, :, :] - mean_origin[1]) / std_origin[1]
    perturbed_image[:, 2, :, :] = (perturbed_image[:, 2, :, :] - mean_origin[2]) / std_origin[2]


    perturbed_image_full[zero_indices[half_idx:idx]] = perturbed_image


    return perturbed_image_full.clone().detach(), stored_image_features.clone().detach(), stored_text_features.clone().detach()

def save_figure_loss(train_loss, val_loss, overall_path, real_part, fake_part, n_ctx):
    # Plot the data
    plt.plot([int(i + 1) for i in range(len(train_loss))], train_loss, label='train loss')
    plt.plot([int(i + 1) for i in range(len(val_loss))], val_loss, label='val loss')

    # Add legend
    plt.legend()

    # Add titles and labels
    plt.title("Train and Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Save the figure to a file
    plt.savefig(f'{overall_path}/Train_Val_Loss_{real_part}_{fake_part}_{n_ctx}.jpg')

    # Show the plot
    plt.clf()


def save_figure_auc(auc_list, overall_path, real_part, fake_part, n_ctx):
    # Plot the data
    # plt.plot([int(i + 1) for i in range(len(accuracy_list['train']))], accuracy_list['train'], label='train_acc')
    plt.plot([int(i + 1) for i in range(len(auc_list['train']))], auc_list['train'], label='train_auc')
    # plt.plot([int(i + 1) for i in range(len(accuracy_list['val']))], accuracy_list['val'], label='val_acc')
    plt.plot([int(i + 1) for i in range(len(auc_list['val']))], auc_list['val'], label='val_auc')

    # Add legend
    plt.legend()

    # Add titles and labels
    plt.title("Train and Val AUC")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")

    # Save the figure to a file
    plt.savefig(f'{overall_path}/ACC_ACU_{real_part}_{fake_part}_{n_ctx}.jpg')

    # Show the plot
    plt.clf()


def get_accuracy(y_binary, gt):
    return np.sum(np.array(gt)==np.array(y_binary))/len(gt)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def text_loss(criterion, features_logits, num_classes, labels):
    # text-axis loss
    # transpose
    labels = labels.t()
    text_feats = features_logits.t()
    tmp_loss = []
    # All images on a prompt: 
    for tmp_class_idx in range(num_classes):
        # mean value of correct prediction
        cur_tmp_loss = [text_feats[tmp_class_idx][labels == tmp_class_idx].mean().unsqueeze(0)]
        for cur_tmp_inner_idx in range(num_classes):
            # mean value of incorrect prediction
            if cur_tmp_inner_idx != tmp_class_idx:
                cur_tmp_loss.append(text_feats[tmp_class_idx][labels == cur_tmp_inner_idx].mean().unsqueeze(0))
        # [T0(I_0123), T1(I_4567)]
        tmp_loss.append(torch.cat(cur_tmp_loss))
    # correct values are indexed 0
    loss = criterion(torch.stack(tmp_loss), torch.zeros(num_classes).long().to(labels.device))

    # # total loss
    # loss = (loss_img + loss_text) / 2 if not torch.isnan(loss_text).any() else loss_img

    return loss

def contrastive_loss(criterion, features_logits, num_classes, labels):
    # Contrastive Loss Begin
    # image-axis loss
    loss_img = criterion(features_logits, labels)

    # text-axis loss
    loss_text = text_loss(criterion, features_logits, num_classes, labels)

    # total loss
    loss = (loss_img + loss_text) / 2 if not torch.isnan(loss_text).any() else loss_img

    return loss

def get_loss(opt, criterion, features_logits, num_classes, labels, prompts_loss):
    loss = criterion(features_logits, labels)

    return loss

def measure_metircs(opt, preds, gt):

    probability = np.array(preds)
    binary_probability = np.zeros_like(probability)

    binary_probability[:, 0] = probability[:, :1].sum(axis=1) #+ probability[:, opt.n_splitNG+opt.n_splitG:].sum(axis=1) # class not G
    binary_probability[:, 1] = probability[:, 1:2].sum(axis=1) # class G

    binary_probability = binary_probability[:, 0:2]

    # binary_probability = torch.nn.functional.softmax(binary_probability, dim=-1)

    final_predictions = np.argmax(binary_probability, axis=1)
    binary_gt = np.array([0 if x < 1 else 1 if x < 2 else 2 for x in gt])

    one_hot_target = np.eye(2)[np.array(binary_gt)]

    # print(one_hot_target.shape)
    # print(binary_probability.shape)

    auc = roc_auc_score(one_hot_target, binary_probability, multi_class='ovr')
    accuracy = accuracy_score(np.array(binary_gt), np.array(final_predictions))
    
    return auc, accuracy

# Define transform for dividing the image into patches and resizing
def patch_image(image, n_patches=2,add_original=False):
    # Get image size
    _, _, height, width = image.size()

    # Calculate patch size
    patch_height, patch_width = height // n_patches, width // n_patches

    patches = []
    for i in range(n_patches):
        #print(i)
        for j in range(n_patches):
            patch = image[:, :, i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width]
            # Resize patch to original size
            patch = interpolate(patch, size=(height, width))
            patches.append(patch)
    if add_original:
        patches.append(image)
    # assert len(patches) ==? 10
    return patches

def seperate_training(opt, criterion, features_logits, num_classes, labels, prompts_loss):
    CLS_weight = opt.CLS_weight
    patch_weight = 1 - CLS_weight

    loss_list = 0

    if len(features_logits.shape) == 2:
        slice_features_logits = features_logits
        temp_loss = get_loss(opt, criterion, slice_features_logits, num_classes, labels, prompts_loss) # 0.8
        # print(weights[i]*temp_loss)
        loss_list = temp_loss

        weighted_logits = features_logits # torch.Size([100, 2])
    elif len(features_logits.shape) == 3:
        for i in range(features_logits.size(1)): #197
            slice_features_logits = features_logits[:, i:i+1, :].squeeze(1)
            # slice_features_logits = features_logits[:, i:i+1, :].squeeze(1) # torch.Size([100, 2])
            temp_loss = get_loss(opt, criterion, slice_features_logits, num_classes, labels, prompts_loss[i]) # 0.8
            # print(weights[i]*temp_loss)
            loss_list += temp_loss
            # loss_list += temp_loss
            # temp_loss_list.append(temp_loss)

        weighted_logits = torch.mean(features_logits, dim=1) # torch.Size([100, 2])

    loss = loss_list

    preds = torch.nn.functional.softmax(weighted_logits, dim=-1)
    # assert preds.size(1) == opt.n_splitNG + opt.n_splitG, f'the number of outputs should be {opt.n_splitNG + opt.n_splitG}'
    
    assert len(preds.shape) == 2, "the dimension of preds should be 2"

    gt = labels.tolist()

    return loss, preds.tolist(), gt

def avarage_probability(opt, criterion, features_logits, num_classes, labels, device, prompts_loss):
    # ############################################################################
    # # Average probability
    # loss_list = 0
    # temp_preds = torch.zeros(features_logits.size(0), 2).to(device)
    # for i in range(1, features_logits.size(1)): # 196
    #     slice_features_logits = features_logits[:, i:i+1, :].squeeze(1)
    #     temp_preds += torch.nn.functional.softmax(slice_features_logits, dim=-1)
    #     temp_loss = get_loss(opt, criterion, slice_features_logits, num_classes, labels)
    #     loss_list += temp_loss
    # loss = loss_list/(features_logits.size(1) - 1)
    # preds += (temp_preds/(features_logits.size(1) - 1)).tolist()
    # gt += labels.tolist()
    # ############################################################################
    ############################################################################
    # Average probability
    CLS_weight = opt.CLS_weight
    patch_weight = 1 - CLS_weight
    loss_list = 0
    CLS_loss = get_loss(opt, criterion, features_logits[:, 0:1, :].squeeze(1), num_classes, labels, prompts_loss[0]) # a number
    CLS_pred = torch.nn.functional.softmax(features_logits[:, 0:1, :].squeeze(1), dim=-1)
    temp_preds = torch.zeros(features_logits.size(0), 2).to(device)
    for i in range(1, features_logits.size(1)):
        slice_features_logits = features_logits[:, i:i+1, :].squeeze(1) # torch.Size([batch, 2])
        temp_preds += torch.nn.functional.softmax(slice_features_logits, dim=-1)
        temp_loss = get_loss(opt, criterion, slice_features_logits, num_classes, labels, prompts_loss[i])
        loss_list += temp_loss
    loss = CLS_weight*CLS_loss + patch_weight*loss_list/(features_logits.size(1) - 1)
    # loss = loss_list/features_logits.size(1)

    preds = (CLS_weight*CLS_pred + patch_weight*temp_preds/(features_logits.size(1) - 1)).tolist()
    gt = labels.tolist()
    ############################################################################

    return loss, preds, gt

def voting(opt, criterion, features_logits, num_classes, labels, device, prompts_loss):
    ############################################################################
    # Voting
    # CLS_weight = 0.8
    # patch_weight = 1 - CLS_weight
    loss_list = 0
    # CLS_loss = get_loss(opt, criterion, features_logits[:, 0:1, :].squeeze(1), num_classes, labels) # a number
    temp_preds = torch.zeros(features_logits.size(0), 2).to(device)
    for i in range(features_logits.size(1)):
        slice_features_logits = features_logits[:, i:i+1, :].squeeze(1)
        max_indices = torch.argmax(slice_features_logits, dim=1)
        temp_preds += F.one_hot(max_indices, num_classes=2)
        temp_loss = get_loss(opt, criterion, slice_features_logits, num_classes, labels, prompts_loss[i])
        loss_list += temp_loss
    # loss = CLS_weight*CLS_loss + patch_weight*loss_list/(features_logits.size(1) - 1)
    loss = loss_list/features_logits.size(1)
    preds = (temp_preds/features_logits.size(1)).tolist()
    gt = labels.tolist()
    ############################################################################

    return loss, preds, gt


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]