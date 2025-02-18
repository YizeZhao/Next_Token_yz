'''
supports training of
1. fixed length next token
2. AR next token
3. ufm
on a relatively small dataset
'''

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
# from functools import partials

import matplotlib.pyplot as plt
import numpy as np

from Dataset import *
from utils import *


import torch
from Model import Transformer, FixedLengthModelArgs, FixedLengthMLP, MultiLabelSVM, TFMModelArgs, UFM
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import json
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train model with configuration file')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
args = parser.parse_args()

# Load configuration from JSON file
with open(args.config, 'r') as f:
    config = json.load(f)

# system
device = config["device"]  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = config["dtype"]  # float32|float16|float16

# dataset
batch_size = config["batch_size"]
if_batch=config["if_batch"]
T = config["T"]

dataset_name = config["dataset_name"]
# traininf_tag = "_cpu_verysmall_ufmtfm_128_fix_s3"
# traininf_tag = "mlp_12_layers_colab0523_run3"
traininf_tag = config["traininf_tag"]

# dataset_name = "verysmallset"
# dataset_name = "tiny_extracted_m100"
tok_file = f'./data/{dataset_name}_pretok_word.bin'
# record_dir = "/Users/yizezhao/Documents/Models/NextToken"
record_dir = config["record_dir"]
# record_dir = "/Users/yizezhao/Desktop/ExpPaperV1"
s_len = config["s_len"]
# set_s = "equal"
# set_s = "random"
set_s = config["set_s"]
# range_repeat = 3
range_repeat = config["range_repeat"]
# save_pretok = None


# model
init_from = config["init_from"] # mlp, tfm or ufm


tokenizer_file = f'./tokenizer/tok_{dataset_name}_word.model'
sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
vocab_size = sp.get_piece_size()

result_dir = f'{record_dir}/results_{dataset_name}{traininf_tag}'
figure_dir = f'{record_dir}/figures_{dataset_name}{traininf_tag}'
for directory in [result_dir, figure_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

bos = 1
eos = 2



# mlp specific
d_hiddens = config["d_hiddens"]
# print(d_hiddens)

# tfm specific
d_encode = config["d_encode"]
# d_hidden = 128
d_decode = config["d_decode"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
multiple_of = config["multiple_of"]
dropout = config["dropout"]
pos_enc = config["pos_enc"]
max_seq_len = config["max_seq_len"]

#ufm specific1
# dim = 512
dim = config["dim"]

# optimizer
lr = config["lr"]
lammy = config["lammy"]
#training
max_iteration = config["max_iteration"]
log_interval = config["log_interval"]

# if solve_by_cvx:
#     lmm_name = "Lmm by cvxpy"
# else:
#     lmm_name = "Lmm by thm"
save_pretok = f'{result_dir}/sampled_{dataset_name}.npy'

if init_from == "ufm":
    if config["balanced_toy"]:
        tsk = Task(batch_size=batch_size,
        T=T, tok_file=tok_file, s_len=s_len, vocab_size=vocab_size, bos=bos, eos=eos,
        device=device, if_ufm=True, x_type=torch.FloatTensor, balanced=True, if_batch=if_batch, set_s=set_s, repeat_range=range_repeat, save_pretok=save_pretok)
    else:
        tsk = Task(batch_size=batch_size,
        T=T, tok_file=tok_file, s_len=s_len, vocab_size=vocab_size, bos=bos, eos=eos,
        device=device, if_ufm=True, x_type=torch.FloatTensor,if_batch=if_batch, set_s=set_s, repeat_range=range_repeat, save_pretok=save_pretok)

else:
    tsk = Task(batch_size=batch_size,
    T=T, tok_file=tok_file, s_len=s_len, vocab_size=vocab_size, bos=bos, eos=eos,
    device=device, if_ufm=False, x_type = torch.LongTensor, if_batch=if_batch, set_s=set_s, repeat_range=range_repeat, save_pretok=save_pretok)# iter_batches = partial(
#     Task.iter_batches, batch_size=batch_size,
#     T=T, tok_file=tok_file, s_len=s_len, vocab_size=vocab_size, bos=bos, eos=eos,
#     device=device,)

support_set_pr = tsk.support_set_pr
support_set_sampled = tsk.support_set_sampled
ctx_dict = tsk.ctx_dict

#train_batch_iter = iter_batches()
tsk_entropy = tsk.emp_entropy
v_ctx = tsk.v_ctx
v_nt = tsk.v_nt
train_batch_iter = tsk.iter_batches()
X = enumerate(train_batch_iter)
# X, Y = X.to(device), Y.to(device)
# print(X.shape, Y.shape)
data_n = tsk.n
data_m = tsk.m
print(f"Dataset m: {data_m}")
print(f"Dataset n: {data_n}")
print(f"Dataset entropy: {tsk_entropy:.4f}")
print(f"Next Token vocab size: {v_nt}")

S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
P = get_p(tsk.ctx_dict, support_set_sampled, support_set_pr, v_nt )


#model_args = dict(T=T, v_ctx=vocab_size, v_nt=vocab_size, d_input=d_encode, d_hidden=d_hidden, d_output=d_decode)

if init_from == "mlp":
    # init a new model from scratch
    print("Initializing a new fixed MLP model from scratch")
    print(d_hiddens)
    #fixedTconf = FixedLengthModelArgs(**model_args)
    model = FixedLengthMLP(T=T, v_ctx=v_ctx, v_nt=v_nt, d_input=d_encode, d_hiddens=d_hiddens)
    uniq_emb = tsk.get_unique()

elif init_from == "tfm":
    print(f'd_decode: {d_decode}')
    model_args = dict(
        dim=d_decode,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=None,
        v_ctx=v_ctx,
        v_nt=v_nt,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pos_enc=pos_enc,)
    print("Initializing a new transformer model from scratch")
    gptconf = TFMModelArgs(**model_args)
    model = Transformer(gptconf)
    uniq_emb = tsk.get_unique()

elif init_from == "ufm":
    # data_m = X.shape[1]
    model = UFM(m=data_m, d=dim, k=v_nt)
    print(f'dim: {dim}')
    print("Initializing a new UFM model from scratch")
    uniq_emb = tsk.get_unique().type(torch.FloatTensor)
    # torch.tensor(np.eye(data_m)).type(torch.FloatTensor)
dummyY = torch.tensor(np.ones(data_m)).type(torch.LongTensor)
dummyY = dummyY.to(device)
uniq_emb = uniq_emb.to(device)
model.to(device)
model_decode_d = model.output.weight.shape[1]
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=lammy)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.geomspace(1, max_iteration, num=5), gamma=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=lammy)
# scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)  # Reduce the learning rate by a factor of 0.1 every 100 epochs
# verysmall
gamma = config["gamma"]
milestones = config["milestones"]
rec_ = config["rec_"]
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
# m100
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 10000,20000, 22500, 25000], gamma=0.25)
# m404
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2500, 4000, 8000,12000,16000, 22500, 25000], gamma=0.5)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1250, 2500, 4000, 8000,12000,16000, 22500, 25000], gamma=0.5)


# uniq_hs = tsk.get_unique()
# if init_from == "ufm":
#     uniq_embeds_dict = model.get_embeddings(tsk.ctx_dict)
# else:
#     uniq_embeds_dict = model.get_embeddings(uniq_emb, tsk.v_ctx2v,tsk.ctx_dict)
# optimizer = torch.optim.Adam(model.parameters())
# all_dict = clean_dict(tsk.ctx_dict, uniq_embeds_dict, support_set_sampled, support_set_pr, tsk.support_set_repeats)
# print(all_dict)
# plot_all_dict(sp, all_dict)
# exit(0)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params} for {model.name} model")

#exit(0)


losses = []
rec_ = [0, 1, 10, 100, 500, 1000]
W_list, H_list, L_list = [],[],[]
W_dir = []
rec, proj_rec = [], []
L_debug, loss_debug = [], []
i = 0
W_f = []
W_star_list, W_fin_list = [],[]
WG_list, HG_list = [], []
train_batch_iter = tsk.iter_batches()
if config["retrain"]:
    for iter_ in range(max_iteration):
        running_loss = 0.0
        dl = tsk.iter_batches()
        for batch_idx, (X, Y) in (enumerate(dl)):
            if if_batch == False:
                X = X[0]
                Y = Y[0]

            X = X.type(tsk.x_type).to(device, non_blocking=True)
            Y = Y.type(torch.LongTensor).to(device, non_blocking=True)
            # debug device
            # print(device)
            # print(X.device, Y.device)
            logits = model(X,Y)
            loss = model.last_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        epoch_loss = running_loss/data_n

        losses.append(epoch_loss)

        if (iter_)%log_interval==0 or iter_ in rec_:
            # This_W = model.output.weight.detach().numpy().copy()
            # w_diff_norm = (This_W - Last_W)/(np.linalg.norm(This_W - Last_W)+1e-5)
            # W_dir.append(w_diff_norm)
            # W_norm.append((model.output.weight.detach().numpy().copy())/W_norm[-1])

            # H_list.append((model.fc1.weight.detach().numpy().copy()))
            with torch.no_grad():
                W_list.append((model.output.weight.cpu().detach().numpy().copy()))
                L_list.append(model(uniq_emb, dummyY).cpu().detach().numpy().copy())
                H_list.append(model.forward_embedding(uniq_emb).cpu().detach().numpy().copy())
                # W_norm.append(np.linalg.norm \
                #                   (model.output.weight.detach().numpy()))
                # L_debug = H_list @

            # Last_W = This_W
            rec.append(iter_)
            print(f"{iter_} | loss {epoch_loss:.4f} | entropy {tsk_entropy:.4f} ")

    # losses = losses
    W_list = np.array(W_list)
    L_list = np.array(L_list)
    H_list = np.array(H_list)

    # save W_list, L_list, H_list
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    np.save(f'{result_dir}/W_list_{model.name}_d{dim}.npy', W_list)
    np.save(f'{result_dir}/L_list_{model.name}_d{dim}.npy', L_list)
    np.save(f'{result_dir}/H_list_{model.name}_d{dim}.npy', H_list)
    np.save(f'{result_dir}/losses_{model.name}_d{dim}.npy', losses)
    np.save(f'{result_dir}/rec_{model.name}_d{dim}.npy', rec)
    np.save(f'{result_dir}/P_{model.name}_d{dim}.npy', P)

    #save the model
    torch.save(model.state_dict(), f'{result_dir}/model_{model.name}.pth')

else:
    # load back  W_list, L_list, H_list
    W_list = np.load(f'{result_dir}/W_list_{model.name}_d{dim}.npy')
    L_list = np.load(f'{result_dir}/L_list_{model.name}_d{dim}.npy')
    H_list = np.load(f'{result_dir}/H_list_{model.name}_d{dim}.npy')
    losses = np.load(f'{result_dir}/losses_{model.name}_d{dim}.npy')
    rec = np.load(f'{result_dir}/rec_{model.name}_d{dim}.npy')

    model.load_state_dict(torch.load(f'{result_dir}/model_{model.name}.pth', map_location=torch.device(device)))

def plot_train():
    nrows, ncols = 2, 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    axs[0].plot(np.arange(len(losses)), losses - tsk_entropy, label='CE loss')
    if config["debug"]:
        axs[0].plot(rec, loss_debug - tsk_entropy, label='debug_loss')
    # axs[0].axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')

    # Add labels and legend
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('loss')
    axs[0].set_yscale('log')
    axs[0].set_title(f'Cross Entropy loss with {model.name} \n on {dataset_name} \n H={tsk_entropy:.4f}')
    axs[0].legend()



    # Show the plot
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if not os.path.exists(f'{figure_dir}'):
        os.makedirs(f'{figure_dir}')
    plt.savefig(f'{figure_dir}/CEloss{model.name}s_'+ str(s_len)+'.pdf')
    # plt.show()






