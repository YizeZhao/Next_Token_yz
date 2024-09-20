import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer
from random import shuffle
from collections import Counter

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task
from export import model_export

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import time 

import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import normalize
import sys
import psutil


from torch.utils.data import DataLoader
from torch.utils.data import Dataset


import seaborn as sns
import matplotlib
sns.set_context("paper")
sns.set_style("whitegrid")
font = {
        'size'   : 18}
matplotlib.rc('font', **font)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# -----------------------------------------------------------------------------
n_heads = 6
n_kv_heads = 6
multiple_of = 32
max_seq_len = 6
dropout = 0.0
pos_enc = "off"
context_length = 6

DATA_PROCESS_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/TinyStories/TinyStories_processing_files"
GRAPH_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/graphs_CameraReady"


from tqdm import tqdm
from tinystories import PretokDataset

vocab_source = "custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained

from tinystories import get_tokenizer_model_path


def compute_structural_component(x, y):
    if x.shape != y.shape:
        raise ValueError("dimensions of x and y do not match")

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = np.std(x, ddof=1)
    sigma_y = np.std(y, ddof=1)

    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    C = 1e-5

    structural_component = (sigma_xy + C) / (sigma_x * sigma_y + C)
    return structural_component


def graph_full_specific_ctxLenght(epochs_heatmap, epochs, context_length, version, num_stories, dim, vocab_size, max_seq_len, AR_training, Norm_Logit, Norm_Feat, ReLU_Feat, On_Test, sort_by = "freq"):
    
    if On_Test:
        ds = PretokDataset("train", max_seq_len, vocab_size, vocab_source, 100, AR_training, Train = False)
    else:
        ds = PretokDataset("train", max_seq_len, vocab_size, vocab_source, 100, AR_training, Train = True)
    

    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)

    target_index = 6 - context_length + 1

    context_to_support_sets = {}
    num_examples = 0
    for X,Y in tqdm(ds):
        context_tuple_full = tuple(X.tolist())
        Y = Y

        context_tuple = context_tuple_full[0:context_length]

        if context_tuple in context_to_support_sets.keys():
            context_to_support_sets[context_tuple]["count"] += 1
            context_to_support_sets[context_tuple]["support"][Y] = 1
        else:
            context_to_support_sets[context_tuple] = {}
            context_to_support_sets[context_tuple]["count"] = 1
            context_to_support_sets[context_tuple]["support"] = np.zeros(vocab_size)
            context_to_support_sets[context_tuple]["support"][Y] = 1
            context_to_support_sets[context_tuple]["tokens"] = enc.sp_model.decode(list(context_tuple))
            context_to_support_sets[context_tuple]["context_tuple_full"] = context_tuple_full
    
        num_examples += 1

    support_counts = {}
    support_set_to_context = {}
    for context in tqdm(list(context_to_support_sets.keys())):
        if tuple(context_to_support_sets[context]["support"]) in set(support_counts.keys()):
            support_counts[tuple(context_to_support_sets[context]["support"])] += 1
            support_set_to_context[tuple(context_to_support_sets[context]["support"])].append(context)
        else:
            support_counts[tuple(context_to_support_sets[context]["support"])] = 1
            support_set_to_context[tuple(context_to_support_sets[context]["support"])] = [context]
    
    sorted_support_counts = {k: v for k, v in sorted(support_counts.items(), key=lambda item: item[1])}

    if sort_by == "sup_freq":
        sorted_support_set_to_context = {k: v for k, v in sorted(support_set_to_context.items(), key=lambda item: len(item[1]))}
    elif sort_by == "sup_size":
        sorted_support_set_to_context = {k: v for k, v in sorted(support_set_to_context.items(), key=lambda item: sum(item[0]))}
    elif sort_by == "both":
        sorted_support_set_to_context = {k: v for k, v in sorted(support_set_to_context.items(), key=lambda item: len(item[1]) + sum(item[0]))}
    

    num_samples_per_sup_set = 10

    list_target_contexts = []
    for sup_set in list(sorted_support_set_to_context.keys())[::-1]:
        if len(list_target_contexts) >= 100:
            break

        sup_set_contexts_list = sorted_support_set_to_context[sup_set]

        if len(sup_set_contexts_list) < num_samples_per_sup_set:
            continue

        sorted_sup_set_contexts_list = sorted(sup_set_contexts_list, key=lambda x: context_to_support_sets[x]['count'])
        for context in sorted_sup_set_contexts_list[::-1][0:num_samples_per_sup_set]:
            list_target_contexts.append(context)

    
    
    X_target = torch.zeros((len(list_target_contexts), 6))
    S_target = torch.zeros((len(list_target_contexts), vocab_size))
    H_target = torch.zeros((len(list_target_contexts), dim))
    Z_target = torch.zeros((len(list_target_contexts), vocab_size))
    Context_Target = []
    for i in range(0, len(list_target_contexts)):
        i_th_context = list_target_contexts[i]
        Context_Target.append(enc.sp_model.decode(list(i_th_context)))
        # X_target[i,:] = torch.tensor(i_th_context)
        X_target[i,:] = torch.tensor(context_to_support_sets[i_th_context]["context_tuple_full"])
        S_target[i,:] = torch.tensor(context_to_support_sets[i_th_context]["support"])
    X_target = X_target.to("cpu").long()
    
    gram_S = S_target @ S_target.T


    ################################################################################################################################################################    
    # log_name = "ColmReb_V" + str(version) + "_L" + str(n_layers) + "_v" + str(vocab_size) + "_s" + str(num_stories) + "_d" + str(dim) +"_lr" + str(learning_rate) + "_total_e" + str(epochs) + "_exp_lrdecay"
    log_name = "ColmCameraReady_V" + str(version) + "_L" + str(n_layers) + "_v" + str(vocab_size) + "_s" + str(num_stories) + "_d" + str(dim) +"_lr" + str(learning_rate) + "_total_e" + str(epochs) + "_exp_lrdecay"
    if AR_training:
        log_name =  log_name + "_ARtrain"
    if ReLU_Feat:
        log_name =  log_name + "_ReLU_Feat"
    if Norm_Feat:
        log_name =  log_name + "_Norm_Feat"
    if Norm_Logit:
        log_name =  log_name + "_Norm_Logit"
    ################################################################################################################################################################    


    for heatmap_epoch in epochs_heatmap:
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)

        dict = torch.load("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/train_files_ColmCameraReady/logs_V" + str(version) + "/" + log_name +  "/model_it" + str(heatmap_epoch) + ".pth", map_location=torch.device('cpu'))
        new_dict = {k[10:]:dict[k] for k in list(dict.keys())}
        model.load_state_dict(new_dict)
        model.eval()
        logits, h = model(X_target, None)


        Z_target = torch.reshape(logits, (len(list_target_contexts), vocab_size)).detach().to("cpu")
        H_target = torch.reshape(h, (len(list_target_contexts), dim)).detach().to("cpu")
        
        gram_Z = Z_target @ Z_target.T
        gram_H = H_target @ H_target.T    
        

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot heatmap for matrix1
        sns.heatmap(gram_S, ax=axs[0], cbar=True)
        axs[0].set_title('S.T @ S')

        # Plot heatmap for matrix2
        sns.heatmap(gram_H, ax=axs[1], cbar=True)
        axs[1].set_title('H.T @ H')

        font = { 'color': 'black', 'weight': 'normal', 'size': 12}
        axs[0].set_xticks([i for i in range(0, 100) if i % 10 == 0], [str(i) for i in range(0, 100) if i % 10 == 0], fontdict=font)
        axs[0].set_yticks([i for i in range(0, 100) if i % 10 == 0], [str(i) for i in range(0, 100) if i % 10 == 0], fontdict=font)

        font = { 'color': 'black', 'weight': 'normal', 'size': 12}
        axs[1].set_xticks([i for i in range(0, 100) if i % 10 == 0], [str(i) for i in range(0, 100) if i % 10 == 0], fontdict=font)
        axs[1].set_yticks([i for i in range(0, 100) if i % 10 == 0], [str(i) for i in range(0, 100) if i % 10 == 0], fontdict=font)


        plt.suptitle("Context Length of " + str(context_length))
        plt.tight_layout()
        if On_Test:
            plt.savefig("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/graphs_CameraReady/Heatmaps/Full_Heatmap_plots" + "_V" + str(version) + "_E" + str(heatmap_epoch) + "_ctxLen" + str(context_length) + "_voc" + str(vocab_size) + "_dim" + str(dim)  + "_numStories" + str(num_stories) + "_sortingFormat_" + sort_by + "_Test.jpg")
        else:
            plt.savefig("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/graphs_CameraReady/Heatmaps/Full_Heatmap_plots" + "_V" + str(version) + "_E" + str(heatmap_epoch) + "_ctxLen" + str(context_length) + "_voc" + str(vocab_size) + "_dim" + str(dim)  + "_numStories" + str(num_stories) + "_sortingFormat_" + sort_by + "_Train.jpg")
        plt.show()
        plt.clf()

    
    ################################################################################################################################################################
    list_of_files = os.listdir("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/train_files_ColmCameraReady/logs_V" + str(version) + "/" + log_name)

    epochs_list = [0]
    for file_name in list_of_files:
        if "model" in file_name:
            epoch = int(file_name.split(".")[0][8:])
            epochs_list.append(epoch)
    epochs_list.sort()
    epochs_list = epochs_list[:-1]

    ################################################################################################################################################################    

    S_matrix = torch.zeros((num_examples, vocab_size))
    X_matrix = torch.zeros((num_examples, context_length), dtype = torch.long)
    index_counter = 0
    context_to_index_dict = {}
    for X,Y in tqdm(ds):
        context_tuple_full = tuple(X.tolist())
        Y = Y

        if context_tuple_full in context_to_index_dict.keys():
            context_index = context_to_index_dict[context_tuple_full]
            S_matrix[context_index, Y] = 1
        else:
            S_matrix[index_counter, Y] = 1
            context_to_index_dict[context_tuple_full] = index_counter
            # if index_counter < 10000:
            X_matrix[index_counter,:] = X
            index_counter += 1
    S_matrix = S_matrix[0:index_counter,:]
    X_matrix = X_matrix[0:index_counter,:]

    S_matrix = S_matrix[0:1000,:]
    X_matrix = X_matrix[0:1000,:]


    # S_target_cosine = torch.nn.functional.normalize(S_matrix, dim = 1)
    # SST_normalized = S_target_cosine[0:index_counter,:] @ S_target_cosine[0:index_counter,:].T
    # SST_normalized = SST_normalized.numpy()

    projection_identity = torch.eye(vocab_size) - (1/vocab_size) * torch.ones((vocab_size,1)) @ torch.ones((1,vocab_size))
    H_G_expected = projection_identity @ S_matrix.T
    H_G_expected_normalized = torch.nn.functional.normalize(H_G_expected, dim = 0)
    H_G_expected_normalized = H_G_expected_normalized.T @ H_G_expected_normalized
    H_G_expected_normalized = H_G_expected_normalized / torch.norm(H_G_expected_normalized, p = 2)
    H_G_expected_normalized = H_G_expected_normalized.numpy()


    projection_identity = torch.eye(vocab_size) - (1/vocab_size) * torch.ones((vocab_size,1)) @ torch.ones((1,vocab_size))
    W_G_expected = projection_identity @ S_matrix.T
    W_G_expected_normalized = torch.nn.functional.normalize(W_G_expected, dim = 1)
    W_G_expected_normalized = W_G_expected_normalized @ W_G_expected_normalized.T
    W_G_expected_normalized = W_G_expected_normalized / torch.norm(W_G_expected_normalized, p = 2)
    W_G_expected_normalized = W_G_expected_normalized.numpy()
    

    H_matrix = torch.zeros((index_counter, dim))
    H_matrix = torch.zeros((1000, dim))

    L_matrix = torch.zeros((index_counter, vocab_size))
    L_matrix = torch.zeros((1000, vocab_size))


    ##########################################
    class MyDataset(Dataset):
        def __init__(self, X_matrix):
            self.data = X_matrix
            
        def __getitem__(self, index):
            x = self.data[index, :]
            return x
        
        def __len__(self):
            return len(self.data)
    
    dataset = MyDataset(X_matrix)
    dl = DataLoader(
        X_matrix,
        batch_size=200,
        num_workers=2,
        shuffle=False
    )
    ##########################################


    H_norm_list = []
    W_norm_list = []
    L_norm_list = []

    SSIM_diff_list_H = []
    SSIM_diff_list_W = []
    for epoch in tqdm(epochs_list):

        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)

        dict = torch.load("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/train_files_ColmCameraReady/logs_V" + str(version) + "/" + log_name +  "/model_it" + str(epoch) + ".pth", map_location=torch.device('cpu'))
        new_dict = {k[10:]:dict[k] for k in list(dict.keys())}
        model.load_state_dict(new_dict)
        model.eval()

        # W Comparison
        W = model.output.weight.clone().detach().to("cpu")
        W_target_cosine = torch.nn.functional.normalize(W, dim = 1)
        WWT_normalized = W_target_cosine @ W_target_cosine.T
        WWT_normalized = WWT_normalized.numpy()
        SSIM_Diff = compute_structural_component(WWT_normalized, W_G_expected_normalized)
        SSIM_diff_list_W.append(SSIM_Diff)
        
        # H Comparison
        counter = 0
        with torch.no_grad():
            for data in dl:
                l, h = model(data, None)
                H_matrix[counter: counter + data.shape[0],:] = h.detach().cpu()
                L_matrix[counter: counter + data.shape[0],:] = l.detach().cpu()
                counter += data.shape[0]

        H_target_cosine = torch.nn.functional.normalize(H_matrix, dim = 1)
        HHT_normalized = H_target_cosine @ H_target_cosine.T
        HHT_normalized = HHT_normalized.numpy()
        SSIM_Diff = compute_structural_component(HHT_normalized, H_G_expected_normalized)
        SSIM_diff_list_H.append(SSIM_Diff)

        # Norms
        H_norm_list.append(torch.norm(H_matrix, p = 2) / H_matrix.shape[0])
        W_norm_list.append(torch.norm(W, p = 2) / vocab_size)
        L_norm_list.append(torch.norm(L_matrix, p = 2) / L_matrix.shape[0])
    

    plt.figure(figsize=(8.3, 4.5))
    plt.plot(epochs_list, np.array(SSIM_diff_list_H), label='TinyStories')
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    # plot in log scale
    plt.xscale('log')
    plt.legend()
    plt.title(f'Structural Correlation between Cos(H, H) and Cos(S.T, S.T), on 100 stories sampled from TinyStories')
    plt.savefig("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/graphs_CameraReady/NewProj_Full_SSIM_H_Conject_plots_" + "e_" + str(epoch) + "_V" + str(version) + "_ctxLen" + str(context_length) + "_voc" + str(vocab_size) + "_dim" + str(vocab_size // 2)  + "_numStories" + str(num_stories) + "_Test.jpg")
    plt.clf()
    # plt.show()


    plt.figure(figsize=(8.3, 4.5))
    plt.plot(epochs_list, np.array(SSIM_diff_list_W), label='TinyStories')
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    # plot in log scale
    plt.xscale('log')
    plt.legend()
    plt.title(f'Structural Correlation between Cos(W, W) and Cos(S, S), on 100 stories sampled from TinyStories')
    plt.savefig("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/graphs_CameraReady/NewProj_Full_SSIM_W_Conject_plots_" + "e_" + str(epoch) + "_V" + str(version) + "_ctxLen" + str(context_length) + "_voc" + str(vocab_size) + "_dim" + str(vocab_size // 2)  + "_numStories" + str(num_stories) + "_Test.jpg")
    plt.clf()
    # plt.show()


    plt.figure(figsize=(8.3, 4.5))
    plt.plot(epochs_list, np.array(H_norm_list), label='||H||')
    plt.plot(epochs_list, np.array(W_norm_list), label='||W||')
    plt.plot(epochs_list, np.array(L_norm_list), label='||L||*')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    # plot in log scale
    plt.yscale('log')
    plt.legend()
    plt.title(f'Norm growth')
    plt.savefig("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/graphs_CameraReady/NewProj_Full_NormGrowth_plots_" + "e_" + str(epoch) + "_V" + str(version) + "_ctxLen" + str(context_length) + "_voc" + str(vocab_size) + "_dim" + str(vocab_size // 2)  + "_numStories" + str(num_stories) + "_Test.jpg")
    plt.clf()
    # plt.show()

On_Test = False
learning_rate = 5e-4
epochs = 600
dim = 512
n_layers = 12
epochs_heatmap = [50, 600]
version = 7 
# vocab_size = 512 # the Llama 2 tokenizer has 32K tokens
num_stories = 100
AR_training = False
max_seq_len = 6
vocab_size = 64

model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
    pos_enc=pos_enc,
    output_dim = dim
)  # start with model_args from command line


context_length = 6
graph_full_specific_ctxLenght(epochs_heatmap, epochs, context_length, version, num_stories, dim, vocab_size, max_seq_len, AR_training, False, False, False, On_Test, sort_by = "sup_size")