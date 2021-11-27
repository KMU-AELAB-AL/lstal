import os
import random
from sklearn.cluster import AgglomerativeClustering

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.datasets import CIFAR100, CIFAR10

from tqdm import tqdm

from config import *
from data.transform import Cifar

from models.resnet import ResNet18
from models.featurenet import FeatureNet

import autoencoder.models.ae as ae
import autoencoder.models.vae as vae

from data.sampler import SubsetSequentialSampler

random.seed('KMU_AELAB')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

transforms = Cifar()

if DATASET == 'cifar10':
    data_train = CIFAR10('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR10('./data', train=True, download=True, transform=transforms.test_transform)
    data_module = CIFAR10('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR10('./data', train=False, download=True, transform=transforms.test_transform)
elif DATASET == 'cifar100':
    data_train = CIFAR100('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR100('./data', train=True, download=True, transform=transforms.test_transform)
    data_module = CIFAR100('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR100('./data', train=False, download=True, transform=transforms.test_transform)


def train_epoch(models, criterions, optimizers, dataloaders, epoch):
    models['backbone'].train()
    models['module'].train()
    models['ae'].eval()

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterions['backbone'](scores, labels)

        if epoch > 120:
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        pred_feature = models['module'](features)
        pred_feature = pred_feature.view([-1, EMBEDDING_DIM])
        ae_out = models['ae'](inputs)

        transfer_loss = torch.mean(1 - criterions['module'](pred_feature, ae_out[1].detach()))

        loss = target_loss + transfer_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()


def test(models, dataloaders, mode='val'):
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


def train(models, criterions, optimizers, schedulers, dataloaders, num_epochs):
    print('>> Train a Model.')

    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        train_epoch(models, criterions, optimizers, dataloaders, epoch)
        if epoch % 10 is 0:
            print(test(models, dataloaders, mode='train'), test(models, dataloaders, mode='test'))

    print('>> Finished.')


def get_uncertainty(models, unlabeled_loader, criterions):
    models['backbone'].eval()
    models['module'].eval()
    models['ae'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            _, features = models['backbone'](inputs)
            pred_feature = models['module'](features)
            pred_feature = pred_feature.view([-1, EMBEDDING_DIM])

            ae_out = models['ae'](inputs)

            loss = 1 - criterions['module'](pred_feature, ae_out[1].detach())

            uncertainty = torch.cat((uncertainty, loss), 0)

    return uncertainty.cpu()


def get_cluster_result(model, data_loader):
    model.eval()

    features = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.cuda()

            ae_out = model(inputs)

            features = torch.cat((features, ae_out[1]), 0)
    features = features.cpu().numpy()

    return AgglomerativeClustering(n_clusters=CLS_CNT).fit_predict(features)


def sampling(cluster_dict):
    sampled = []
    while len(sampled) < ADDENDUM:
        order = sorted(list(cluster_dict.keys()), key=lambda x: cluster_dict[x])
        quo, remain = (ADDENDUM - len(sampled)) // len(order), (ADDENDUM - len(sampled)) % len(order)
        quo = quo if quo < min([len(i) for i in cluster_dict.values()]) else min([len(i) for i in cluster_dict.values()])

        if quo:
            for idx in order:
                random.shuffle(cluster_dict[idx])
                sampled.extend(cluster_dict[idx][:quo])
                cluster_dict[idx] = cluster_dict[idx][quo:]

                if not len(cluster_dict[idx]):
                    del cluster_dict[idx]
        else:
            for idx in order[:remain]:
                sampled.append(cluster_dict[idx].pop)


if __name__ == '__main__':
    target_module = vae.VAE(NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS, EMBEDDING_DIM)
    checkpoint = torch.load(f'trained_ae/vae_{DATASET}.pth.tar')
    target_module.load_state_dict(checkpoint['ae_state_dict'])
    target_module.cuda()

    for trial in range(TRIALS):
        fp = open(f'record_{trial + 1}.txt', 'w')

        indices = list(range(NUM_TRAIN))
        clustered_label = get_cluster_result(target_module, DataLoader(data_unlabeled, batch_size=BATCH,
                                                                       sampler=SubsetSequentialSampler(indices),
                                                                       pin_memory=True))

        random.shuffle(indices)
        labeled_set = indices[:INIT_CNT]
        unlabeled_set = indices[INIT_CNT:]

        train_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        dataloaders = {'train': train_loader, 'test': test_loader}

        resnet18 = ResNet18(num_classes=CLS_CNT).cuda()
        feature_module = FeatureNet(out_dim=EMBEDDING_DIM).cuda()

        models = {'backbone': resnet18, 'module': feature_module, 'ae': target_module}

        torch.backends.cudnn.benchmark = False

        for cycle in range(CYCLES):
            criterion = nn.CrossEntropyLoss().cuda()
            m_criterion = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
            criterions = {'backbone': criterion, 'module': m_criterion}

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = optim.Adam(models['module'].parameters(), lr=1e-3)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module = lr_scheduler.ReduceLROnPlateau(optim_module, mode='min', factor=0.8, cooldown=4)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            train(models, criterions, optimizers, schedulers, dataloaders, EPOCH)
            acc = test(models, dataloaders, mode='test')

            fp.write(f'{acc}\n')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))

            unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(unlabeled_set),
                                          pin_memory=True)

            uncertainty = get_uncertainty(models, unlabeled_loader, criterions)

            arg = np.argsort(uncertainty)

            subset = list(torch.tensor(unlabeled_set)[arg][-SUBSET:].numpy())
            subset_label = clustered_label[subset]
            subset_cluster = {}
            for idx in range(SUBSET):
                if subset_label[idx] not in subset_cluster:
                    subset_cluster[subset_label[idx]] = [subset[idx]]
                else:
                    subset_cluster[subset_label[idx]].append(subset[idx])

            labeled_set += sampling(subset_cluster)
            unlabeled_set = list(set(unlabeled_set) - set(labeled_set))

            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)
