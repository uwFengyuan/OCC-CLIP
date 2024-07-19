import json, time, random, os, GPUtil
from utils import patch_image, seperate_training, \
    avarage_probability, voting, get_loss, measure_metircs, fgsm_attack
from dataloader import model_aware_load

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(opt, model, criterion, optimizer, scheduler, n_epochs, device, overall_path, transform, batch):
    
    best_val_loss, best_train_loss = float('inf'), 0
    train_loss_list, valid_loss_list = [], []
    train_time_list, val_time_list = [], []
    auc_list, accuracy_list = {'train': [], 'val': []}, {'train': [], 'val': []}
    best_epoch = 0
    best_auc, best_accuracy = 0, 0
    num_classes = opt.n_classes

    train_dataset = model_aware_load(opt.train_target,'train', opt.train_real, opt, transform)
    valid_dataset = model_aware_load(opt.train_target,'val', opt.train_real, opt, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=True)

    real_part = ','.join(opt.train_real)
    target_part = ','.join(opt.train_target)

    with open(f'{overall_path}/best_{opt.model}_{opt.backbone}_{opt.n_ctx}_{real_part}_{target_part}.txt', 'a') as f:
        f.write(f'Parameters are :{opt}\n')
        f.write(f'Transform: {transform}\n')
        f.write(f'Criterion: {criterion}\n')
        f.write(f'Optimizer: {optimizer}\n')
        f.write(f'Scheduler: {scheduler.state_dict()}\n')

    start_time = time.time()

    Image_Features_O_list = []
    Image_Features_P_list = []
    Text_Features_list = []

    for epoch in range(n_epochs):
        preds = []
        gt = []
        model.train()  # Set model to training mode
        train_loss = 0.0
        for original_inputs, original_labels in tqdm(train_loader):
            inputs, labels = original_inputs.to(device), original_labels.to(device)
            
            if opt.random_epsilon:
                numbers = [0.03, 0.1, 0.5]  # replace with your list of numbers
                epsilon = random.choice(numbers)
            else:
                epsilon = opt.epsilon

            perturbed_inputs, image_features_O, text_features = fgsm_attack(model, inputs, opt, criterion, num_classes, labels, epsilon)

            if opt.batch >= 50:
                zero_indices = (labels == 0).nonzero().squeeze()
                one_indices = (labels == 1).nonzero().squeeze()

                image_features_original = []
                image_features_original.extend(image_features_O[zero_indices].detach().cpu().numpy().tolist())
                image_features_original.extend(image_features_O[one_indices].detach().cpu().numpy().tolist())

                Image_Features_O_list.append(image_features_original)
                Text_Features_list.append(text_features.detach().cpu().numpy().tolist())

            # if opt.shuffled:
            shuffle_idx_p = torch.randperm(perturbed_inputs.size(0))
            perturbed_inputs = perturbed_inputs[shuffle_idx_p]
            labels = labels[shuffle_idx_p]


            # Training on adversarial examples
            optimizer.zero_grad()
            features_logits, image_features_P, _ = model(perturbed_inputs)
            prompts_loss = torch.zeros(1).to(features_logits.device)

            if opt.batch >= 50:
                zero_indices_p = (labels == 0).nonzero().squeeze()
                one_indices_p = (labels == 1).nonzero().squeeze()
                
                image_features_perturbed = []
                image_features_perturbed.extend(image_features_P[zero_indices_p].detach().cpu().numpy().tolist())
                image_features_perturbed.extend(image_features_P[one_indices_p].detach().cpu().numpy().tolist())

                Image_Features_P_list.append(image_features_perturbed)  

            if len(features_logits.shape) == 2:
                temp_features_logits = features_logits
                preds += torch.nn.functional.softmax(temp_features_logits, dim=1).tolist()
            elif len(features_logits.shape) == 3:
                temp_features_logits = features_logits[:, 0:1, :].squeeze(1)
                preds += torch.nn.functional.softmax(temp_features_logits, dim=1).tolist()
            gt += labels.tolist()   

            loss = get_loss(opt, criterion, temp_features_logits, num_classes, labels, prompts_loss)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        mean_train_loss = train_loss / len(train_loader.dataset)
        middle_time = time.time()
        time_taken1 = middle_time - start_time
        train_time_list.append(time_taken1)
        train_auc, train_accuracy = measure_metircs(opt, preds, gt)
        
        train_auc = max(train_auc, 1 - train_auc)
        train_accuracy = max(train_accuracy, 1- train_accuracy)


        if epoch % 5 == 0:
            
            mean_valid_loss, val_auc, val_accuracy, time_taken2 = \
                validation(model, opt, None, None, valid_loader, device, criterion, num_classes, None, overall_path)

            tempLR = scheduler.get_lr()[0]
            print(f"Epoch: {epoch+1}, LR: {tempLR:.6f}, Epsilon:{epsilon:.2f}, Training Loss: {mean_train_loss:.4f}, Validation Loss: {mean_valid_loss:.4f}, AUC_T: {train_auc:.4f}, ACC_T: {train_accuracy:.4f}, AUC_V: {val_auc:.4f}, ACC_V: {val_accuracy:.4f}")

            accuracy_list['val'].append(val_accuracy)
            auc_list['val'].append(val_auc)
            valid_loss_list.append(mean_valid_loss)
            val_time_list.append(time_taken2)

            accuracy_list['train'].append(train_accuracy)
            auc_list['train'].append(train_auc)
            train_loss_list.append(mean_train_loss)

            best_val_loss = mean_valid_loss
            best_train_loss = mean_train_loss
            best_epoch = epoch
            best_auc = val_auc
            best_accuracy = val_accuracy
        
        scheduler.step()

    # save and visualize image features
    if opt.batch >= 50:
        with open(f'{overall_path}/Image_Features_P_{opt.model}_{opt.backbone}_{opt.n_ctx}_{real_part}_{target_part}.json', 'w') as f:
            json.dump(Image_Features_P_list, f)
        with open(f'{overall_path}/Image_Features_O_{opt.model}_{opt.backbone}_{opt.n_ctx}_{real_part}_{target_part}.json', 'w') as f:
            json.dump(Image_Features_O_list, f)
        with open(f'{overall_path}/Text_Features_{opt.model}_{opt.backbone}_{opt.n_ctx}_{real_part}_{target_part}.json', 'w') as f:
            json.dump(Text_Features_list, f)
        
    torch.save(model.state_dict(), f'{overall_path}/best_{opt.model}_{opt.backbone}_{opt.n_ctx}_{real_part}_{target_part}.pt')
    with open(f'{overall_path}/results_train.txt', 'a') as f:
        f.write(f'Model is :{opt.model}_{opt.backbone}_{opt.n_ctx}_{real_part}_{target_part}. Epochs: {len(train_time_list)}\n')
        f.write(f'Train time consuming: total ({np.sum(train_time_list)}), per epoch ({np.mean(train_time_list)})\n')
        f.write(f'Val time consuming: total ({np.sum(val_time_list)}), per epoch ({np.mean(val_time_list)})\n')
        f.write(f"Epoch: {best_epoch+1}, Training Loss: {best_train_loss:.4f}, Validation Loss: {best_val_loss:.4f}, AUC: {best_auc:.4f}, ACC: {best_accuracy:.4f}\n")

    total_loss = { "train_loss": train_loss_list, "val_loss": valid_loss_list}
    with open(f'{overall_path}/Loss_{opt.model}_{opt.backbone}_{opt.n_ctx}_{real_part}_{target_part}.json', 'w') as f:
       json.dump(total_loss, f)

    return train_loss_list, valid_loss_list, real_part, target_part, accuracy_list, auc_list

def validation(model, opt, zero_value, one_value, data_loader, device, criterion, num_classes, model_path, overall_path):
    if opt.test:
        model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    valid_loss = 0.0
    preds = []
    gt = []
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            features_logits, _, _ = model(inputs)
            prompts_loss = torch.zeros(1).to(features_logits.device)

            if len(features_logits.shape) == 2:
                temp_features_logits = features_logits
                preds += torch.nn.functional.softmax(temp_features_logits, dim=1).tolist()
            elif len(features_logits.shape) == 3:
                temp_features_logits = features_logits[:, 0:1, :].squeeze(1)
                preds += torch.nn.functional.softmax(temp_features_logits, dim=1).tolist()

            gt += labels.tolist()  
            if not opt.test: 
                loss = get_loss(opt, criterion, temp_features_logits, num_classes, labels, prompts_loss)

            if not opt.test: 
                valid_loss += loss.item() * inputs.size(0)

    time_taken = time.time() - start_time       
    auc, accuracy = measure_metircs(opt, preds, gt)
    auc = max(auc, 1 - auc)
    accuracy = max(accuracy, 1- accuracy)

    if opt.test:
        mean_valid_loss = 0
        
        print(f'AUC: {auc:.4f}, ACC: {accuracy}')


        if not os.path.exists(overall_path):
            os.makedirs(overall_path)
        with open(f'{overall_path}/{opt.model}_{opt.backbone}_{opt.n_ctx}_{zero_value}_{one_value}_preds.json', 'w') as f:
            json.dump(preds, f)

    else:
        mean_valid_loss = valid_loss / len(data_loader.dataset)

    return mean_valid_loss, auc, accuracy, time_taken

def task(model, opt, criterion, overall_path, device, transform):
    
    train_real = opt.train_real
    train_target = opt.train_target
    tested_name = opt.test_other
    test_target = opt.test_target
    
    if not isinstance(tested_name, list):
        tested_name = [tested_name]

    assert isinstance(tested_name, list), "Object is not a list."

    test_dataset = model_aware_load(test_target, 'test', tested_name, opt, transform)
    zero_value = ','.join(tested_name)
    one_value = ','.join(test_target)

    test_loader = DataLoader(test_dataset, batch_size=1024)
    real_part = ','.join(train_real)
    train_target_part = ','.join(train_target)

    model_path = os.path.join(overall_path, f'best_{opt.model}_{opt.backbone}_{opt.n_ctx}_{real_part}_{train_target_part}.pt')
    model_result_path = f"{real_part}_{train_target_part}"
    task_path = os.path.join(model_result_path, ','.join(test_target))
    task_path = os.path.join(overall_path, task_path)
    _, auc, accuracy, test_time_taken = validation(model, opt, zero_value, one_value, test_loader, \
                                        device, criterion, 2, model_path, task_path)


    with open(os.path.join(overall_path, f'results_{model_result_path}.txt'), 'a') as f:
        f.write(f'{opt.model}_{opt.backbone}_{opt.n_ctx}: Model is :{model_path}.\n')
        f.write(f'Test datasets are from {zero_value}+{one_value}. AUC: {auc:.4f}, ACC: {accuracy} \n')
        f.write(f'Test time consuming: {test_time_taken}\n')
        f.write('\n')