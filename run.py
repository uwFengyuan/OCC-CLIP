import os.path as osp

from utils import ConstantWarmupScheduler, save_figure_loss, save_figure_auc
import os
import torchvision.transforms as tt
from ADA import task, train
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from models import CoOp
import torch, argparse, os
import torch.nn as nn

import random
import numpy as np
import wandb


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="CoOp",help="choose which model to run")
    parser.add_argument("-b", "--backbone", type=str, default='ViT', choices=['RN50', 'ViT', 'NoBB'], 
                         help="choose which backbone to run")
    parser.add_argument("--n_ctx",type=int,default=16, help="number of ctx used in experiments",)
    parser.add_argument("--train",action='store_true')
    parser.add_argument("--train_real",nargs="+",
                        help="enter the name of open source real dataset for training",)
    parser.add_argument("--train_target",nargs="+",
                        help="enter the name of target dataset for training",)
    parser.add_argument("--test",action='store_true')
    parser.add_argument("--test_other",nargs="+",
                        help="enter the name of another dataset for testing",)
    parser.add_argument("--test_target",nargs="+",
                        help="enter the name of target dataset for testing",)
    parser.add_argument("--n_classes",type=int,default=2, help="number of data used in experiments",)
    parser.add_argument("--seed",type=int,default=1, help="number of data used in experiments",)
    parser.add_argument("--n_epoch", type=int,default=200) 
    parser.add_argument("--train_set",type=int, default=1)
    parser.add_argument("--lr",type=float,default=0.0001,)
    parser.add_argument("--save_path",type=str,default='', help="Save trained model and test results")
    parser.add_argument("--epsilon",type=float,default=0.1,)
    parser.add_argument("--batch",type=int,default=10, help="batch size for training",)
    parser.add_argument("--Use_Attack",action='store_true',help="Use Adversarial Data Augmentation (ADA)",)
    parser.add_argument("--random_epsilon",action='store_true',help="step length of ADA",)
    parser.add_argument("--n_shots",type=int,default=50,help="number of images for training",)
    parser.add_argument("--pos_name",type=str,default='fake')
    parser.add_argument("--neg_name",type=str,default='real')
    parser.add_argument("--divideNum",type=float,default=0.5,help="proportion of adversarial number",)


    opt = parser.parse_args()
    random_seed = opt.seed
    random.seed(random_seed)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.set_printoptions(precision=8)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


    if opt.Use_Attack:
        use_attack = ""
        if opt.random_epsilon:
            use_attack += 'UseAttackRandom'
        else:
            use_attack += 'UseAttack' + str(opt.epsilon)
    else:
        use_attack = 'NoAttack'
        opt.epsilon = 0.0

    batch = opt.batch
    # transform_image = opt.transform_image

    model, transform = CoOp(opt)

    print("Turning off gradients in both the image and the text encoder")
    layers_to_check = ["prompt_learner", "output_layer"]
    for name, param in model.named_parameters():
        if any(s in name for s in layers_to_check):
            param.requires_grad_(True)
        # if "prompt_learner" not in name or "output_layer" not in name or "fc" not in name:
        else:
            param.requires_grad_(False)

    learning_rate = opt.lr #0.001

    # print(f"Transform: {transform_image}")
    print(transform)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Params: %.2f' % (params))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)        

    overall_path = opt.save_path
    overall_path = os.path.join(overall_path, "Shots" + str(opt.n_shots) + f"_{opt.model}{opt.backbone}_{use_attack}")
    overall_path = os.path.join(overall_path, "NEG" + ','.join(opt.train_real) + "_POS" + ','.join(opt.train_target) + "_seed" + str(opt.seed))
    overall_path = os.path.join(overall_path, "TrainSet" + str(opt.train_set) + "_" + str(opt.neg_name) + "_" + str(opt.pos_name))

    if not os.path.exists(overall_path):
        os.makedirs(overall_path)
    print(f'Path: {overall_path}')

    criterion = nn.CrossEntropyLoss()

    if opt.train:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, dampening=0, weight_decay=0.0005, nesterov=False)
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        scheduler = ConstantWarmupScheduler(optimizer, cos_scheduler, 1, 1e-5)

        n_epochs = opt.n_epoch

        # Train and evaluate
        train_loss, val_loss, real_part, fake_part, _ , auc_list = train(opt, model, criterion, \
                    optimizer, scheduler, n_epochs, device, overall_path, transform, batch)

        save_figure_loss(train_loss, val_loss, overall_path, real_part, fake_part, opt.n_ctx)
        save_figure_auc(auc_list, overall_path, real_part, fake_part, opt.n_ctx)

        import json

        with open(f'{overall_path}/train_loss.json', 'w') as file:
            json.dump(train_loss, file)

        with open(f'{overall_path}/val_loss.json', 'w') as file:
            json.dump(val_loss, file)

        with open(f'{overall_path}/auc_list.json', 'w') as file:
            json.dump(auc_list, file)
            
    if opt.test:
        task(model, opt, criterion, overall_path, device, transform)

    


if __name__ == "__main__":
    main()  
