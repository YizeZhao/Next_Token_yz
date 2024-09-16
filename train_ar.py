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

from Dataset import *
from utils import *


import torch
from Model import Transformer, FixedLengthModelArgs, FixedLengthMLP, MultiLabelSVM, TFMModelArgs, UFM
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


debug = False
proj_each = False
do_svm, check_wmm = False, False
balanced_toy = False

# TODO: CHECK arguments before running
train = True
resume = False
keep_cnt = 5
solve_by_cvx = False
check_lmm = True
# random.seed(1634)
# np.random.seed(1634)

# system
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("CUDA Current Device:", torch.cuda.current_device())
print("CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "float32"  # float32|float16|float16

# dataset
batch_size = 64
if_batch=True
T = 6

# dataset_name = "tiny_extract_m404"
# traininf_tag = "_cpu_verysmall_ufmtfm_128_fix_s3"
traininf_tag = "_colab_positionloss_0501_newlr"
# dataset_name = "verysmallset"
# dataset_name = "tiny_extracted_m100"
dataset_name = "tiny_2000lines"
# dataset_name = "tiny100lines"
tok_file = f'./data/{dataset_name}_pretok_char.bin'
# record_dir = "/Users/yizezhao/Documents/Models/NextToken"
record_dir = "./colab_results"
# record_dir = "/Users/yizezhao/Desktop/ExpPaperV1"
s_len = 3
set_s = "equal"
# set_s = "random"
# set_s = "original"
# range_repeat = 3
range_repeat = 5
# save_pretok = None


# model
init_from = "tfm" # mlp, tfm or ufm


tokenizer_file = f'./tokenizer/tok_{dataset_name}_char.model'
sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
vocab_size = sp.get_piece_size()

result_dir = f'{record_dir}/results_{dataset_name}{traininf_tag}'
figure_dir = f'{record_dir}/figures_{dataset_name}{traininf_tag}'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

bos = 1
eos = 2



# mlp specific
d_encode = 128
d_hidden = 128
d_decode = 128

# tfm specific
dim = 256
n_layers = 8
n_heads = 6
n_kv_heads = 6
multiple_of = 4
dropout = 0.0
pos_enc = "off"
max_seq_len = 16

#ufm specific1
# dim = 512

# optimizer
lr = 1e-3
lammy = 1e-6
#training
max_iteration = 200
log_interval = 20
save_model_interval = 50

# if solve_by_cvx:
#     lmm_name = "Lmm by cvxpy"
# else:
#     lmm_name = "Lmm by thm"
save_pretok = f'{result_dir}/sampled_{dataset_name}.npy'

ds = ARDataset(max_seq_len, vocab_size, tok_file)
dl = torch.utils.data.DataLoader(
    ds, batch_size=batch_size, pin_memory=True, num_workers=0
)




#model_args = dict(T=T, v_ctx=vocab_size, v_nt=vocab_size, d_input=d_encode, d_hidden=d_hidden, d_output=d_decode)

# if init_from == "mlp":
#     # init a new model from scratch
#     print("Initializing a new fixed MLP model from scratch")
#     #fixedTconf = FixedLengthModelArgs(**model_args)
#     model = FixedLengthMLP(T=T, v_ctx=v_ctx, v_nt=v_nt, d_input=d_encode, d_hidden=d_hidden, d_output=d_decode)
#     uniq_emb = tsk.get_unique()


if init_from == "tfm":
    model_args = dict(
        dim=d_decode,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=None,
        v_ctx=vocab_size,
        v_nt=vocab_size,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pos_enc=pos_enc,
        if_ar=True,)
    print("Initializing a new transformer model from scratch")
    gptconf = TFMModelArgs(**model_args)
    model = Transformer(gptconf)
    start_i = 0
    losses, positional_losses = [], []
    rec_ = [0, 1, 10, 100, 500, 1000]
    # W_list, H_list, L_list = [],[],[]
    rec = []

    if resume:
        # find the latest model file in the result directory and store the iteration number
        model_files = [f for f in os.listdir(result_dir) if f.endswith('.pth')]
        if len(model_files) > 0:
            model_files.sort()
            model_file = model_files[-1]
            model.load_state_dict(torch.load(f'{result_dir}/{model_file}'))
            # start_i = int(model_file.split('_')[-1].split('.')[0]) + 1
            losses = list(np.load(f'{result_dir}/losses_{model.name}_d{dim}.npy'))
            start_i = len(losses)
            positional_losses = list(np.load(f'{result_dir}/positional_losses_{model.name}_d{dim}.npy'))
            rec = list(np.load(f'{result_dir}/rec_{model.name}_d{dim}.npy'))
            # print(rec)
            if keep_cnt > 0:
                losses = losses[:keep_cnt]
                positional_losses = positional_losses[:keep_cnt]
                rec = rec[:keep_cnt]
                #start_i = int(model_file.split('_')[-1].split('.')[0]) + 1
                start_i = 51
            print(f"Resuming training from {model_file} at iteration {start_i}")



model.to(device)
model_decode_d = model.output.weight.shape[1]
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=lammy)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.geomspace(1, max_iteration, num=5), gamma=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=lammy)
# scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)  # Reduce the learning rate by a factor of 0.1 every 100 epochs
# verysmall
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 10000,15000,20000,25000], gamma=0.4)
# m100
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 10000,20000, 22500, 25000], gamma=0.25)
# m404
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2500, 4000, 8000,12000,16000, 22500, 25000], gamma=0.6)




if train:
    for iter_ in range(start_i, max_iteration):
        running_loss, position_running_loss = 0.0, np.zeros(max_seq_len)
        data_n = 0
        for batch_idx, (X, Y) in (enumerate(dl)):
            data_n += Y.shape[0]
            if if_batch == False:
                X = X[0]
                Y = Y[0]
            # device = "cuda"
            X = X.to(device, dtype=torch.long)
            Y = Y.to(device, dtype=torch.long)
            # print(X.device, Y.device, next(model.parameters()).is_cuda)
            logits = model(X,Y)
            position_loss = model.positionwise_loss
            loss = model.last_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            position_running_loss += position_loss.cpu().detach().numpy()

        epoch_loss = running_loss/data_n
        position_epoch_loss = position_running_loss/data_n

        if iter_ in rec_ or (iter_)%log_interval==0:
            print(f"Iter {iter_} | Loss {epoch_loss:.4f} | Position Loss {[f'{x:.4f}' for x in position_epoch_loss[[0, 5, 10, 15]]]}")
            rec.append(iter_)
            # record losses and positional losses
            losses.append(epoch_loss)
            positional_losses.append(position_epoch_loss)

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            losses_np = np.array(losses)
            positional_losses_np = np.array(positional_losses)
            np.save(f'{result_dir}/losses_{model.name}_d{dim}.npy', losses_np)
            np.save(f'{result_dir}/positional_losses_{model.name}_d{dim}.npy', positional_losses_np)
            # save rec
            np.save(f'{result_dir}/rec_{model.name}_d{dim}.npy', rec)
            torch.save(model.state_dict(), f'{result_dir}/model_{model.name}_int.pth')

        if iter_%save_model_interval==0:
            # save model
            torch.save(model.state_dict(), f'{result_dir}/model_{model.name}_{iter_}.pth')


    # save W_list, L_list, H_list

    torch.save(model.state_dict(), f'{result_dir}/model_{model.name}_{iter_}.pth')
    print('Finished Training')

def plot_losses():

    # losses = np.load(f'{result_dir}/losses_{model.name}_d{dim}.npy')
    # positional_losses = np.load(f'{result_dir}/positional_losses_{model.name}_d{dim}.npy')
    # #reconstruct rec
    # rec = sorted(list(set(rec_+list(range(0, max_iteration-1, log_interval))))[:len(losses)])
    # plot losses and positional losses, positional losses has dimension (len(rec), max_seq_len)
    
    losses = np.array(losses)
    positional_losses = np.array(positional_losses)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(rec, losses)
    ax[0].set_title(f'Losses {model.name}')
    for index, y in enumerate(positional_losses.T):
        ax[1].plot(rec, y, label=f'position {index}')
    ax[0].legend()
    ax[1].legend()
    plt.savefig(f'{figure_dir}/losses_{model.name}_d{dim}.pdf')
    print(f'Losses plot saved at {figure_dir}/losses_{model.name}_d{dim}.pdf')
    plt.close()

plot_losses()