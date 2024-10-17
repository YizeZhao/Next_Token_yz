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
from scipy.linalg import svd, diagsvd
import matplotlib.gridspec as gridspec



import torch
from Model import Transformer, FixedLengthModelArgs, FixedLengthMLP, MultiLabelSVM, TFMModelArgs, UFM, LSTMModel
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


debug = False
proj_each = False
do_svm, check_wmm = False, False
balanced_toy = False

# TODO: CHECK arguments before running
retrain = True
solve_by_cvx = False
check_lmm = False
# random.seed(1634)
# np.random.seed(1634)

device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "float32"  # float32|float16|float16

# dataset
batch_size = 32
if_batch=False
T = 6

dataset_name = "tiny_extract_m404"
# dataset_name = "verysmallset"

# traininf_tag = "_cpu_verysmall_tfm_128_fix_s3"
# traininf_tag = "_cpu_ufm128_404_sori"
model_name = "lstm"
#model_name = "4 layer Transformer pos off"
# dataset_name = "tiny_extracted_m100"
tok_file = f'./data/{dataset_name}_pretok_word.bin'
record_dir = "/Users/Lenovo/Next_Token/cpu_results"

if dataset_name == "verysmallset":
    s_len = 3
    set_s = "equal"
    range_repeat = 3

elif dataset_name == "tiny_extract_m404":
    s_len = 3
    set_s = "original"
    range_repeat = 5

# model
init_from = "lstm" # mlp, tfm or ufm

load_tf404 = True
load_tfsmall = False
load_ufm404 = True
load_ufmsmall = False
load_lstmsmall = False
load_lstm404 = True

ufm_verysmall_result = f"{record_dir}/results_verysmallset_cpu_verysmall_ufm_128_fix_s3"
tfm_verysmall_result = f"{record_dir}/results_verysmallset_cpu_verysmall_tfm_128_fix_s3"
ufm_tiny404_result = f"{record_dir}/results_tiny_extract_m404_cpu_ufm128_404_sori"
# tfm_tiny404_result = f"{record_dir}/results_tiny_extract_m404_colab_tfm128_404_sori"
tfm_tiny404_result = f"{record_dir}/results_tiny_extract_m404_cpu_404_tfm_test1"
lstm_tiny404_result = f"{record_dir}/results_tiny_extract_m404_cpu_404_lstm_test1"
lstm_verysmall_result = f"{record_dir}/results_verysmallset_cpu_404_lstm_test1"

if model_name == "ufm":
    if dataset_name == "verysmallset":
        result_dir = ufm_verysmall_result
    else:
        result_dir = ufm_tiny404_result
elif model_name == "lstm":
    if dataset_name == "verysmallset":
        result_dir = lstm_verysmall_result
    else:
        result_dir = lstm_tiny404_result
else :
    if dataset_name == "verysmallset":
        result_dir = tfm_verysmall_result
    else:
        result_dir = tfm_tiny404_result


tokenizer_file = f'./tokenizer/tok_{dataset_name}_word.model'
sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
vocab_size = sp.get_piece_size()

# result_dir = ufm_verysmall_result

figure_dir = f'{record_dir}/exp_figures_addLSTM'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

bos = 1
eos = 2

# mlp specific
d_hiddens = [1024]*5 + [512] * 5 + [256] * 5 + [128] * 5
# print(d_hiddens)

# tfm specific
d_encode = 128
# d_hidden = 128
d_decode = 128
n_layers = 8
n_heads = 6
n_kv_heads = 6
multiple_of = 4
dropout = 0.0
pos_enc = "off"
max_seq_len = 6
extra_ln = 128

#ufm specific1
# dim = 512
dim = 128

# optimizer
lr = 1e-4
lammy = 1e-6
#training
max_iteration = 12000
log_interval = 1000

# if solve_by_cvx:
#     lmm_name = "Lmm by cvxpy"
# else:
#     lmm_name = "Lmm by thm"

save_pretok = f'{result_dir}/sampled_{dataset_name}.npy'


if init_from == "ufm":
    if balanced_toy:
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
    device=device, if_ufm=False, x_type = torch.LongTensor, if_batch=if_batch, set_s=set_s, repeat_range=range_repeat, save_pretok=save_pretok)
# iter_batches = partial(
#     Task.iter_batches, batch_size=batch_size,
#     T=T, tok_file=tok_file, s_len=s_len, vocab_size=vocab_size, bos=bos, eos=eos,
#     device=device,)

support_set_pr = tsk.support_set_pr
support_set_sampled = tsk.support_set_sampled
ctx_dict = tsk.ctx_dict

#train_batch_iter = iter_batches()
tsk_entropy = tsk.emp_entropy
print(f"Dataset entropy: {tsk_entropy}")
v_ctx = tsk.v_ctx
v_nt = tsk.v_nt
# train_batch_iter = tsk.iter_batches()
# X, Y = next(train_batch_iter)
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

# exit(0)
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
        pos_enc=pos_enc,
        if_extra_ln=extra_ln)
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

elif init_from == "lstm":
    print("Initializing a new LSTM model")
    # Initialize the LSTM model
    model = LSTMModel(T=T, v_ctx=v_ctx, v_nt=v_nt, d_input=d_encode, d_hiddens=d_hiddens)
    uniq_emb = tsk.get_unique()

dummyY = torch.tensor(np.ones(data_m)).type(torch.LongTensor)
dummyY = dummyY.to(device)
uniq_emb = uniq_emb.to(device)
model.to(device)

W_list_init = ((model.output.weight.cpu().detach().numpy().copy()))
L_list_init = (model(uniq_emb, dummyY).cpu().detach().numpy().copy())
H_list_init = (model.forward_embedding(uniq_emb).cpu().detach().numpy().copy())

#model_args = dict(T=T, v_ctx=vocab_size, v_nt=vocab_size, d_input=d_encode, d_hidden=d_hidden, d_output=d_decode)

# load back  W_list, L_list, H_list
W_list = np.load(f'{result_dir}/W_list_{model_name}_d{dim}.npy')
L_list = np.load(f'{result_dir}/L_list_{model_name}_d{dim}.npy')
H_list = np.load(f'{result_dir}/H_list_{model_name}_d{dim}.npy')
losses = np.load(f'{result_dir}/losses_{model_name}_d{dim}.npy')
rec = np.load(f'{result_dir}/rec_{model_name}_d{dim}.npy')
# P = np.load(f'{result_dir}/P_{model_name}_d{dim}.npy')
# S = np.zeros_like(P)
# S[P>0] = 1
# model.load_state_dict(torch.load(f'{result_dir}/model_{model.name}.pth', map_location=torch.device(device)))

H_1000 = np.load(f'{result_dir}/H_list_{model_name}_d{dim}_1000.npy')
W_1000 = np.load(f'{result_dir}/W_list_{model_name}_d{dim}_1000.npy')
L_1000 = np.load(f'{result_dir}/L_list_{model_name}_d{dim}_1000.npy')

# merge two lists
W_list = np.concatenate((W_1000[0:4,:,:], W_list[1:,:,:]), axis=0)
L_list = np.concatenate((L_1000[0:4,:,:], L_list[1:,:,:]), axis=0)
H_list = np.concatenate((H_1000[0:4,:,:], H_list[1:,:,:]), axis=0)
rec = np.concatenate((np.array([0, 1, 10, 100]), rec[1:]), axis=0)

if load_tf404:
    W_list_tf404 = np.load(f'{tfm_tiny404_result}/W_list_4 layer Transformer pos off_d128.npy')
    L_list_tf404 = np.load(f'{tfm_tiny404_result}/L_list_4 layer Transformer pos off_d128.npy')
    H_list_tf404 = np.load(f'{tfm_tiny404_result}/H_list_4 layer Transformer pos off_d128.npy')
    losses_tf404 = np.load(f'{tfm_tiny404_result}/losses_4 layer Transformer pos off_d128.npy')
    rec_tf404 = np.load(f'{tfm_tiny404_result}/rec_4 layer Transformer pos off_d128.npy')
    P_tf404 = np.load(f'{tfm_tiny404_result}/P_4 layer Transformer pos off_d128.npy')
    S_tf404 = np.zeros_like(P_tf404)
    S_tf404[P_tf404>0] = 1

    W_list_tf404_1000 = np.load(f'{tfm_tiny404_result}/W_list_4 layer Transformer pos off_d128_1000.npy')
    L_list_tf404_1000 = np.load(f'{tfm_tiny404_result}/L_list_4 layer Transformer pos off_d128_1000.npy')
    H_list_tf404_1000 = np.load(f'{tfm_tiny404_result}/H_list_4 layer Transformer pos off_d128_1000.npy')

    W_list_tf404 = np.concatenate((W_list_tf404_1000[0:4,:,:], W_list_tf404[1:,:,:]), axis=0)
    L_list_tf404 = np.concatenate((L_list_tf404_1000[0:4,:,:], L_list_tf404[1:,:,:]), axis=0)
    H_list_tf404 = np.concatenate((H_list_tf404_1000[0:4,:,:], H_list_tf404[1:,:,:]), axis=0)
    rec_tf404 = np.concatenate((np.array([0, 1, 10, 100]), rec_tf404[1:]), axis=0)


if load_tfsmall:
    W_list_tfsmall = np.load(f'{tfm_verysmall_result}/W_list_4 layer Transformer pos off_d128.npy')
    L_list_tfsmall = np.load(f'{tfm_verysmall_result}/L_list_4 layer Transformer pos off_d128.npy')
    H_list_tfsmall = np.load(f'{tfm_verysmall_result}/H_list_4 layer Transformer pos off_d128.npy')
    losses_tfsmall = np.load(f'{tfm_verysmall_result}/losses_4 layer Transformer pos off_d128.npy')
    rec_tfsmall = np.load(f'{tfm_verysmall_result}/rec_4 layer Transformer pos off_d128.npy')
    P_tfsmall = np.load(f'{tfm_verysmall_result}/P_4 layer Transformer pos off.npy')
    S_tfsmall = np.zeros_like(P_tfsmall)
    S_tfsmall[P_tfsmall>0] = 1

    W_list_tfsmall_1000 = np.load(f'{tfm_verysmall_result}/W_list_4 layer Transformer pos off_d128_1000.npy')
    L_list_tfsmall_1000 = np.load(f'{tfm_verysmall_result}/L_list_4 layer Transformer pos off_d128_1000.npy')
    H_list_tfsmall_1000 = np.load(f'{tfm_verysmall_result}/H_list_4 layer Transformer pos off_d128_1000.npy')

    W_list_tfsmall = np.concatenate((W_list_tfsmall_1000[0:4,:,:], W_list_tfsmall[1:,:,:]), axis=0)
    L_list_tfsmall = np.concatenate((L_list_tfsmall_1000[0:4,:,:], L_list_tfsmall[1:,:,:]), axis=0)
    H_list_tfsmall = np.concatenate((H_list_tfsmall_1000[0:4,:,:], H_list_tfsmall[1:,:,:]), axis=0)
    rec_tfsmall = np.concatenate((np.array([0, 1, 10, 100]), rec_tfsmall[1:]), axis=0)



if load_ufm404:
    W_list_ufm404 = np.load(f'{ufm_tiny404_result}/W_list_UFM_d128.npy')
    L_list_ufm404 = np.load(f'{ufm_tiny404_result}/L_list_UFM_d128.npy')
    H_list_ufm404 = np.load(f'{ufm_tiny404_result}/H_list_UFM_d128.npy')
    losses_ufm404 = np.load(f'{ufm_tiny404_result}/losses_UFM_d128.npy')
    rec_ufm404 = np.load(f'{ufm_tiny404_result}/rec_UFM_d128.npy')
    # P_ufm404 = np.load(f'{ufm_tiny404_result}/P_UFM_d128.npy')
    # S_ufm404 = np.zeros_like(P)
    # S_ufm404[P>0] = 1
    W_list_ufm404_1000 = np.load(f'{ufm_tiny404_result}/W_list_UFM_d128_1000.npy')
    L_list_ufm404_1000 = np.load(f'{ufm_tiny404_result}/L_list_UFM_d128_1000.npy')
    H_list_ufm404_1000 = np.load(f'{ufm_tiny404_result}/H_list_UFM_d128_1000.npy')

    W_list_ufm404 = np.concatenate((W_list_ufm404_1000[0:4,:,:], W_list_ufm404[1:,:,:]), axis=0)
    L_list_ufm404 = np.concatenate((L_list_ufm404_1000[0:4,:,:], L_list_ufm404[1:,:,:]), axis=0)
    H_list_ufm404 = np.concatenate((H_list_ufm404_1000[0:4,:,:], H_list_ufm404[1:,:,:]), axis=0)
    rec_ufm404 = np.concatenate((np.array([0, 1, 10, 100]), rec_ufm404[1:]), axis=0)

if load_ufmsmall:
    W_list_ufmsmall = np.load(f'{ufm_verysmall_result}/W_list_UFM_d128.npy')
    L_list_ufmsmall = np.load(f'{ufm_verysmall_result}/L_list_UFM_d128.npy')
    H_list_ufmsmall = np.load(f'{ufm_verysmall_result}/H_list_UFM_d128.npy')
    losses_ufmsmall = np.load(f'{ufm_verysmall_result}/losses_UFM_d128.npy')
    rec_ufmsmall = np.load(f'{ufm_verysmall_result}/rec_UFM_d128.npy')
    # P_ufmsmall = np.load(f'{ufm_verysmall_result}/P_UFM_d128.npy')
    # S_ufmsmall = np.zeros_like(P)
    # S_ufmsmall[P>0] = 1

    W_list_ufmsmall_1000 = np.load(f'{ufm_verysmall_result}/W_list_UFM_d128_1000.npy')
    L_list_ufmsmall_1000 = np.load(f'{ufm_verysmall_result}/L_list_UFM_d128_1000.npy')
    H_list_ufmsmall_1000 = np.load(f'{ufm_verysmall_result}/H_list_UFM_d128_1000.npy')

    W_list_ufmsmall = np.concatenate((W_list_ufmsmall_1000[0:4,:,:], W_list_ufmsmall[1:,:,:]), axis=0)
    L_list_ufmsmall = np.concatenate((L_list_ufmsmall_1000[0:4,:,:], L_list_ufmsmall[1:,:,:]), axis=0)
    H_list_ufmsmall = np.concatenate((H_list_ufmsmall_1000[0:4,:,:], H_list_ufmsmall[1:,:,:]), axis=0)
    rec_ufmsmall = np.concatenate((np.array([0, 1, 10, 100]), rec_ufmsmall[1:]), axis=0)

if load_lstm404:
    W_list_lstm404 = np.load(f'{lstm_tiny404_result}/W_list_LSTM_d128.npy')
    L_list_lstm404 = np.load(f'{lstm_tiny404_result}/L_list_LSTM_d128.npy')
    H_list_lstm404 = np.load(f'{lstm_tiny404_result}/H_list_LSTM_d128.npy')
    losses_lstm404 = np.load(f'{lstm_tiny404_result}/losses_LSTM_d128.npy')
    rec_lstm404 = np.load(f'{lstm_tiny404_result}/rec_LSTM_d128.npy')
    P_lstm404 = np.load(f'{lstm_tiny404_result}/P_LSTM.npy')
    S_lstm404 = np.zeros_like(P_tf404)
    S_lstm404[P_tf404>0] = 1

    W_list_lstm404_1000 = np.load(f'{lstm_tiny404_result}/W_list_LSTM_d128_1000.npy')
    L_list_lstm404_1000 = np.load(f'{lstm_tiny404_result}/L_list_LSTM_d128_1000.npy')
    H_list_lstm404_1000 = np.load(f'{lstm_tiny404_result}/H_list_LSTM_d128_1000.npy')

    W_list_lstm404 = np.concatenate((W_list_lstm404_1000[0:4,:,:], W_list_lstm404[1:,:,:]), axis=0)
    L_list_lstm404 = np.concatenate((L_list_lstm404_1000[0:4,:,:], L_list_lstm404[1:,:,:]), axis=0)
    H_list_lstm404 = np.concatenate((H_list_lstm404_1000[0:4,:,:], H_list_lstm404[1:,:,:]), axis=0)
    rec_lstm404 = np.concatenate((np.array([0, 1, 10, 100]), rec_lstm404[1:]), axis=0)

def plot_train():
    nrows, ncols = 2, 1
    fig, axs = plt.subplots(nrows, ncols,  dpi=100)

    axs[0].plot(np.arange(len(losses)), losses - tsk_entropy, label='CE loss')

    # Add labels and legend
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('loss')
    axs[0].set_yscale('log')
    axs[0].set_title(f'Cross Entropy loss with {model_name} \n on {dataset_name} \n H={tsk_entropy:.4f}')
    axs[0].legend()



    # Show the plot
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if not os.path.exists(f'{figure_dir}'):
        os.makedirs(f'{figure_dir}')
    plt.savefig(f'{figure_dir}/CEloss{model_name}s_'+ str(s_len)+'.pdf')
    print(f"saved in {figure_dir}/CEloss{model_name}s_{s_len}")
    # plt.show()

# plot_rec_samples = [0,1,2,3,4,5,6,7,10,15,20,30]
# compute rank of W, L, H
def plot_losses():
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, dpi=100, figsize=(6, 4))

    tsk_entropy_404 = 1.6597
    tsk_entropy_small = 1.0166

    print(losses_lstm404[-1])
    # axs.plot(range(len(losses_ufm404)), losses_ufm404 - tsk_entropy_404,'b--', label='Simplified TinyStories, UFM')
    # axs.plot(range(len(losses_tf404)), losses_tf404 - tsk_entropy_404,'b-', label='Simplified TinyStories, TF')
    # axs.plot(range(len(losses_tfsmall)), losses_tfsmall - tsk_entropy_small, 'r-',label='Synthetic, TF')
    # axs.plot(range(len(losses_ufmsmall)), losses_ufmsmall - tsk_entropy_small,'r--', label='Synthetic , UFM')
    # # axs[0].axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')
    #
    # # Add labels and legend
    # axs.set_xlabel('Epoch')
    # # axs.set_ylabel('loss - entropy')
    # axs.set_yscale('log')
    # axs.set_xscale('log')
    # # axs.set_title(f'Cross Entropy loss with {model_name} \n on {dataset_name} \n H={tsk_entropy:.4f}')
    # # axs.legend()
    # # Creating custom legend entries
    # custom_lines = [
    #     Line2D([0], [0,1], color='r',),
    #     Line2D([0], [0,1], color='b',),
    #     Line2D([0], [0,1], color='k', linestyle='-'),
    #     Line2D([0], [0,1], color='k', linestyle='--')
    # ]
    #
    # custom_labels = [
    #     'Synthetic',
    #     'Simplified TinyStories',
    #     'TF',
    #     'UFM'
    # ]


    # colors
    axs.grid()
    axs.plot(range(len(losses_ufm404)), losses_ufm404 - tsk_entropy_404,color='deepskyblue', label='Simplified TinyStories, UFM')
    axs.plot(range(len(losses_tf404)), losses_tf404 - tsk_entropy_404, color='steelblue', label='Simplified TinyStories, TF')
    #axs.plot(range(len(losses_tfsmall)), losses_tfsmall - tsk_entropy_small, color='darkorange',label='Synthetic, TF')
    #axs.plot(range(len(losses_ufmsmall)), losses_ufmsmall - tsk_entropy_small,'gold', label='Synthetic , UFM')
    # axs[0].axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')
    axs.plot(range(len(losses_lstm404)), losses_lstm404 - tsk_entropy_404,color='green', label='Simplified TinyStories, LSTM')

    # Add labels and legend
    axs.set_xlabel(r'Epoch $k$')
    # axs.set_ylabel('loss - entropy')
    axs.set_yscale('log')
    axs.set_xscale('log')

    # axs.set_title(f'Cross Entropy loss with {model_name} \n on {dataset_name} \n H={tsk_entropy:.4f}')
    # axs.legend()
    # Creating custom legend entries
    custom_lines = [
        Line2D([0], [0,1], color='gold',),
        Line2D([0], [0,1], color='darkorange',),
        Line2D([0], [0,1], color='deepskyblue'),
        Line2D([0], [0,1], color='steelblue'),
        Line2D([0], [0,1], color='green')
    ]

    custom_labels = [
        'Synthetic, UFM',
        'Synthetic, TF',
        'Simplified TinyStories, UFM',
        'Simplified TinyStories, TF',
        'Simplified TinyStories, LSTM'
    ]

    # plt.legend(custom_lines, custom_labels, title='Legend Title')
    plt.legend(custom_lines, custom_labels)


    # Show the plot
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if not os.path.exists(f'{figure_dir}'):
        os.makedirs(f'{figure_dir}')
    plt.savefig(f'{figure_dir}/CElosses_all_{model_name}_{dataset_name}_d{dim}.pdf')
    # plt.show()

plot_losses()
# exit(0)

def plot_norms():
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, dpi=100, figsize=(6, 4))

    W_norm_tf_404 = np.linalg.norm(W_list_tf404, axis=(1, 2))
    H_norm_tf_404 = np.linalg.norm(H_list_tf404, axis=(1, 2))

    # W_norm_tf_small = np.linalg.norm(W_list_tfsmall, axis=(1, 2))
    # H_norm_tf_small = np.linalg.norm(H_list_tfsmall, axis=(1, 2))

    W_norm_ufm_404 = np.linalg.norm(W_list_ufm404, axis=(1, 2))
    H_norm_ufm_404 = np.linalg.norm(H_list_ufm404, axis=(1, 2))

    # W_norm_ufm_small = np.linalg.norm(W_list_ufmsmall, axis=(1, 2))
    # H_norm_ufm_small = np.linalg.norm(H_list_ufmsmall, axis=(1, 2))

    W_norm_lstm_404 = np.linalg.norm(W_list_lstm404, axis=(1, 2))
    H_norm_lstm_404 = np.linalg.norm(H_list_lstm404, axis=(1, 2))

    n_small = 72
    n_404 = 5110
    v_small = 30
    v_404 = 104

    # # plot all
    # mksz = 5
    # axs.plot(rec_tf404[plot_rec_samples]+1, W_norm_tf_404[plot_rec_samples],'b-o',markersize=mksz, label='Simplified TinyStories, TF, ||W_t||')
    # axs.plot(rec_tf404[plot_rec_samples]+1, H_norm_tf_404[plot_rec_samples], 'b-x',markersize=mksz, label='Simplified TinyStories, TF, ||H_t||')
    # axs.plot(rec_tfsmall[plot_rec_samples]+1, W_norm_tf_small[plot_rec_samples], 'r-o',markersize=mksz,  label='Synthetic , TF, ||W_t||')
    # axs.plot(rec_tfsmall[plot_rec_samples]+1, H_norm_tf_small[plot_rec_samples], 'r-x',markersize=mksz, label='Synthetic , TF, ||H_t||')
    # axs.plot(rec_ufm404[plot_rec_samples]+1, W_norm_ufm_404[plot_rec_samples], 'b--o',markersize=mksz, label='Simplified TinyStories, UFM, ||W_t||')
    # axs.plot(rec_ufm404[plot_rec_samples]+1, H_norm_ufm_404[plot_rec_samples], 'b--x',markersize=mksz, label='Simplified TinyStories, UFM, ||H_t||')
    # axs.plot(rec_ufmsmall[plot_rec_samples]+1, W_norm_ufm_small[plot_rec_samples], 'r--o',markersize=mksz, label='Synthetic , UFM, ||W_t||')
    # axs.plot(rec_ufmsmall[plot_rec_samples]+1, H_norm_ufm_small[plot_rec_samples], 'r--x',markersize=mksz, label='Synthetic , UFM, ||H_t||')
    #
    # # axs[0].set_title('Norm')
    # # axs.set_xlabel('iterations')
    # axs.set_yscale('log')
    # # set x axis range
    # axs.set_xscale('log')
    # axs.set_xlim(1, 30000)
    # #
    # # axs.set_title(f'Cross Entropy loss with {model_name} \n on {dataset_name} \n H={tsk_entropy:.4f}')
    # # axs.legend()
    # # Creating custom legend entries
    # custom_lines = [
    #     Line2D([0], [0,1], color='r',),
    #     Line2D([0], [0,1], color='b',),
    #     Line2D([0], [0,1], color='k', linestyle='-'),
    #     Line2D([0], [0,1], color='k', linestyle='--'),
    #     Line2D([0], [0, 1], color='k', marker='o'),
    #     Line2D([0], [0, 1], color='k', marker='x')
    # ]
    #
    # custom_labels = [
    #     'Synthetic',
    #     'Simplified TinyStories',
    #     'TF',
    #     'UFM',
    #     'W',
    #     'H'
    # ]

    # colors
    mksz = 5
    axs.plot(rec_tf404+1, W_norm_tf_404, color='steelblue',linestyle='--', label='Simplified TinyStories, TF, ||W_t||')
    axs.plot(rec_tf404+1, H_norm_tf_404, color='steelblue', label='Simplified TinyStories, TF, ||H_t||')
    # axs.plot(rec_tfsmall+1, W_norm_tf_small, color='darkorange',linestyle='--',  label='Synthetic , TF, ||W_t||')
    # axs.plot(rec_tfsmall+1, H_norm_tf_small, color='darkorange', label='Synthetic , TF, ||H_t||')
    axs.plot(rec_ufm404+1, W_norm_ufm_404, color='deepskyblue',linestyle='--', label='Simplified TinyStories, UFM, ||W_t||')
    axs.plot(rec_ufm404+1, H_norm_ufm_404, color='deepskyblue', label='Simplified TinyStories, UFM, ||H_t||')
    # axs.plot(rec_ufmsmall+1, W_norm_ufm_small, color='gold',linestyle='--',label='Synthetic , UFM, ||W_t||')
    # axs.plot(rec_ufmsmall+1, H_norm_ufm_small, color='gold', label='Synthetic , UFM, ||H_t||')
    axs.plot(rec_lstm404+1, W_norm_lstm_404, color='green',linestyle='--', label='Simplified TinyStories, LSTM, ||W_t||')
    axs.plot(rec_lstm404+1, H_norm_lstm_404, color='green', label='Simplified TinyStories, LSTM, ||H_t||')

    # axs[0].set_title('Norm')
    axs.set_xlabel(r'Epoch $k$')
    axs.set_yscale('log')
    # set x axis range
    axs.set_xscale('log')
    axs.set_xlim(1, 30000)
    axs.grid()
    #
    # axs.set_title(f'Cross Entropy loss with {model_name} \n on {dataset_name} \n H={tsk_entropy:.4f}')
    # axs.legend()
    # Creating custom legend entries

    custom_lines = [
        Line2D([0], [0,1], color='gold',),
        Line2D([0], [0,1], color='darkorange',),
        Line2D([0], [0,1], color='deepskyblue'),
        Line2D([0], [0,1], color='steelblue'),
        Line2D([0], [0,1], color='green'),
        Line2D([0], [0, 1], color='k', linestyle='-'),
        Line2D([0], [0, 1], color='k', linestyle='--'),
    ]

    custom_labels = [
        'Synthetic, UFM',
        'Synthetic, TF',
        'Simplified TinyStories, UFM',
        'Simplified TinyStories, TF',
        'Simplified TinyStories, LSTM',
        # '||H||',
        # '||W||'
        'H',
        'W'
    ]


    # plt.legend(custom_lines, custom_labels, title='Legend Title')
    plt.legend(custom_lines, custom_labels)

    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(f'{figure_dir}/WHnormgrowth_all_{dataset_name}_{model_name}_d{dim}.pdf')
    # plt.show()

plot_norms()
# exit(0)



# W star direction convergence
#     check
#     1. Proj_f(W) -> W_fin
#     2. W->W_star
L_list_normed = L_list/(np.linalg.norm(L_list, axis=(1,2) ))[:, np.newaxis, np.newaxis]
W_list_normed = W_list/(np.linalg.norm(W_list, axis=(1,2) ))[:, np.newaxis, np.newaxis]
H_list_normed = H_list/(np.linalg.norm(H_list, axis=(1,2) ))[:, np.newaxis, np.newaxis]
W_dir = W_list_normed[1:] - W_list_normed[:-1]
L_dir = L_list_normed[1:] - L_list_normed[:-1]
H_dir = H_list_normed[1:] - H_list_normed[:-1]

W_norm = np.linalg.norm(W_list, axis=(1,2) )
H_norm = np.linalg.norm(H_list, axis=(1,2) )
W_dir_norm = np.linalg.norm(W_dir, axis=(1,2) )
L_dir_norm = np.linalg.norm(L_dir, axis=(1,2))
H_dir_norm = np.linalg.norm(H_dir, axis=(1,2))




# exit(0)

# Lmm, GW
# if init_from == "ufm" or init_from == "tfm":
# S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
# if solve_by_cvx:
# L_mm_cvxpy = SVM_cvxpy_sol(S)
# load lmm cvxpy
if dataset_name == "verysmallset":
    L_mm_cvxpy = np.load(f'{ufm_verysmall_result}/Lmm_cvxpy_verysmallset.npy')
else:
    L_mm_cvxpy = np.load(f'{ufm_tiny404_result}/Lmm_cvxpy_{dataset_name}.npy')

L_mm_cvxpy_normed = L_mm_cvxpy / np.linalg.norm(L_mm_cvxpy)

# save lmm cvxpy
# np.save(f'{result_dir}/Lmm_cvxpy_{dataset_name}.npy', L_mm_cvxpy)
# else:
L_mm_thm = SVM_thm_sol(tsk.ctx_dict, support_set_sampled, v_nt)
np.save(f'{result_dir}/Lmm_thm_{dataset_name}.npy', L_mm_thm)
# exit(0)
# L_mm_thm = SVM_thm_sol2(S,v_nt)
L_mm_thm_normed = L_mm_thm / np.linalg.norm(L_mm_thm)
# l_mm_normed = L_mm / np.linalg.norm(L_mm)

GL_mm_cvxpy = np.dot(L_mm_cvxpy, L_mm_cvxpy.T)
GL_mm_thm = np.dot(L_mm_thm, L_mm_thm.T)
GL_mm_cvxpy_T = np.dot(L_mm_cvxpy.T, L_mm_cvxpy)
GL_mm_thm_T = np.dot(L_mm_thm.T, L_mm_thm)

sqrt_GL_mm_cvxpy = matrix_sqrt(GL_mm_cvxpy)
sqrt_GL_mm_thm = matrix_sqrt(GL_mm_thm)
sqrt_GL_mm_cvxpy_T = matrix_sqrt(GL_mm_cvxpy_T)
sqrt_GL_mm_thm_T = matrix_sqrt(GL_mm_thm_T)

L_fin = compute_L_fin(P, S)
L_proj_L_fin_diff = project_Lt(P, S, L_list, L_fin)
L_train_L_fin = L_list - L_fin
L_train_L_fin_norm = np.linalg.norm(L_train_L_fin, axis=(1,2))
L_list_norm = np.linalg.norm(L_list, axis=(1,2))
L_train_L_fin_norm_ratio = L_train_L_fin_norm / L_list_norm
print("L_train_L_fin_norm_ratio:", L_train_L_fin_norm_ratio)
print("L fin norm:", np.linalg.norm(L_fin, 'fro'))
print("L list norm:", np.linalg.norm(L_list, axis=(1,2)))
L_train_L_fin_Lmm_cvxpy = compute_norm_corr(L_train_L_fin, L_mm_cvxpy_normed)
L_train_L_fin_Lmm_thm = compute_norm_corr(L_train_L_fin, L_mm_thm_normed)

print("rank of S:", np.linalg.matrix_rank(S))
print("rank of Lmm cvxpy:", np.linalg.matrix_rank(L_mm_cvxpy))
print("rank of Lmm thm:", np.linalg.matrix_rank(L_mm_thm))

print("rank of S:", get_rank_by_svd(S))
print("rank of Lmm cvxpy:", get_rank_by_svd(L_mm_cvxpy))
print("rank of Lmm thm:", get_rank_by_svd(L_mm_thm))

# exit(0)

def plot_L_proj_L_fin_diff():
    # compute L_fin for both datasets
    L_fin_tf404 = compute_L_fin(P_tf404, S_tf404)
    L_fin_tfsmall = compute_L_fin(P_tfsmall, S_tfsmall)
    # compute L_proj_L_fin_diff for all four models
    L_proj_L_fin_diff_tf404 = project_Lt(P_tf404, S_tf404, L_list_tf404, L_fin_tf404)
    L_proj_L_fin_diff_tfsmall = project_Lt(P_tfsmall, S_tfsmall, L_list_tfsmall, L_fin_tfsmall)
    L_proj_L_fin_diff_ufm404 = project_Lt(P_tf404, S_tf404, L_list_ufm404, L_fin_tf404)
    L_proj_L_fin_diff_ufmsmall = project_Lt(P_tfsmall, S_tfsmall, L_list_ufmsmall, L_fin_tfsmall)
    L_proj_L_fin_diff_lstm404 = project_Lt(P_tf404, S_tf404, L_list_lstm404, L_fin_tf404)
    #L_proj_L_fin_diff_lstmsmall = project_Lt(P_tfsmall, S_tfsmall, L_list_lstmsmall, L_fin_tfsmall)

    # plot all four L_proj_L_fin_diff
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, dpi=100, figsize=(6, 4)) #, sharex=True, sharey=True

    # axs.plot(rec_tf404, L_proj_L_fin_diff_tf404, 'b-', label='Simplified TinyStories, TF')
    # axs.plot(rec_tfsmall, L_proj_L_fin_diff_tfsmall, 'r-', label='Synthetic, TF')
    # axs.plot(rec_ufm404, L_proj_L_fin_diff_ufm404, 'b--', label='Simplified TinyStories, UFM')
    # axs.plot(rec_ufmsmall, L_proj_L_fin_diff_ufmsmall, 'r--', label='Synthetic, UFM')
    #
    # # Add labels and legend
    # axs.set_xlabel('Epoch')
    # # axs.set_ylabel('||L_t - L_fin||')
    # axs.set_yscale('log')
    # axs.set_xscale('log')
    # # axs.set_title(f'Cross Entropy loss with {model_name} \n on {dataset_name} \n H={tsk_entropy:.4f}')
    # custom_lines = [
    #     Line2D([0], [0,1], color='r',),
    #     Line2D([0], [0,1], color='b',),
    #     Line2D([0], [0,1], color='k', linestyle='-'),
    #     Line2D([0], [0,1], color='k', linestyle='--')
    # ]
    #
    # custom_labels = [
    #     'Synthetic',
    #     'Simplified TinyStories',
    #     'TF',
    #     'UFM'
    # ]

    axs.plot(rec_tf404, L_proj_L_fin_diff_tf404, color='steelblue', label='Simplified TinyStories, TF')
    axs.plot(rec_tfsmall, L_proj_L_fin_diff_tfsmall, color='darkorange', label='Synthetic, TF')
    axs.plot(rec_ufm404, L_proj_L_fin_diff_ufm404,  color='deepskyblue', label='Simplified TinyStories, UFM')
    axs.plot(rec_ufmsmall, L_proj_L_fin_diff_ufmsmall, color='gold', label='Synthetic, UFM')
    axs.plot(rec_lstm404, L_proj_L_fin_diff_lstm404,  color='green', label='Simplified TinyStories, LSTN')
    # Add labels and legend
    axs.set_xlabel(r'Epoch $k$')
    # axs.set_ylabel('||L_t - L_fin||')
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.grid()

    custom_lines = [
        Line2D([0], [0, 1], color='gold', ),
        Line2D([0], [0, 1], color='darkorange', ),
        Line2D([0], [0, 1], color='deepskyblue'),
        Line2D([0], [0, 1], color='steelblue'),
        Line2D([0], [0, 1], color='green')
    ]

    custom_labels = [
        'Synthetic, UFM',
        'Synthetic, TF',
        'Simplified TinyStories, UFM',
        'Simplified TinyStories, TF',
        'Simplified TinyStories, LSTM'
    ]
    plt.legend(custom_lines, custom_labels)
    # Show the plot
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if not os.path.exists(f'{figure_dir}'):
        os.makedirs(f'{figure_dir}')
    plt.savefig(f'{figure_dir}/L_proj_L_fin_diff_all_{model_name}_{dataset_name}_d{dim}.pdf')
    # plt.show()

# curve
# plot_L_proj_L_fin_diff()
# exit(0)



print("computing grams for Lmm cvxpy")
WG_mm_norm_cvxpy, HG_mm_norm_cvxpy, W_mm_cvxpy, H_mm_cvxpy, u_cvxpy, vh_cvxpy, WG_mm_cvxpy, HG_mm_cvxpy = compute_grams(L_mm_cvxpy, v_nt)
print("computing grams for Lmm thm")
WG_mm_norm_thm, HG_mm_norm_thm, W_mm_thm, H_mm_thm, u_thm, vh_thm, WG_mm_thm, HG_mm_thm = compute_grams(L_mm_thm, v_nt)

def compare_logits():
    # generate and save a plot with all 3 matrices on the same colorbar
    # print difference
    print("dataset:", dataset_name)
    print("Lmm cvxpy 2-norm:", np.linalg.norm(L_mm_cvxpy))
    print("Lmm thm 2-norm:", np.linalg.norm(L_mm_thm))
    print("Lmm cvxpy nuclear norm:", np.linalg.norm(L_mm_cvxpy, 'nuc'))
    print("Lmm thm nuclear norm:", np.linalg.norm(L_mm_thm, 'nuc'))
    print("avg difference per entry:", np.mean(np.abs(L_mm_cvxpy - L_mm_thm)))
    nrows, ncols = 1, 3
    # fsize=22
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    im1 = axs[2].imshow(L_mm_cvxpy.T, cmap='magma', interpolation='nearest', vmin=-1, vmax=1)
    # axs[2].set_title('Lmm cvxpy', )
    im2 = axs[1].imshow(L_mm_thm.T, cmap='magma', interpolation='nearest', vmin=-1, vmax=1)
    # axs[1].set_title('Lmm thm', )
    im3 = axs[0].imshow(S.T, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    # axs[0].set_title('Support Set',)
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.colorbar(im3)

    plt.savefig(f'{figure_dir}/S_L_mm_cvxpy_thm_{dataset_name}.pdf')

# compare_logits()
# exit(0)



def plot_logits_verysmallsets():
    L_list_ufm = np.load(
        f'{ufm_verysmall_result}/L_list_UFM_d128.npy')
    L_list_tfm = np.load(
        f'{tfm_verysmall_result}/L_list_4 layer Transformer pos off_d128.npy')
    # rec = np.load(f'/Users/Lenovo/Next_Token/ExpPaperV1/results_tiny_extract_m404_cpu_ufm128_404_sori/rec_UFM_d128.npy')
    L_ufm = L_list_ufm[-1]
    L_ufm_normed = L_ufm / np.linalg.norm(L_ufm)
    L_tfm = L_list_tfm[-1]
    L_tfm_normed = L_tfm / np.linalg.norm(L_tfm)
    # imshow L_ufm, L_tfm and L_mm_cvxpy on the same colorbar
    # color_max = np.max([np.max(L_ufm_normed), np.max(L_tfm_normed), np.max(L_mm_cvxpy_normed)])
    # color_min = np.min([np.min(L_ufm_normed), np.min(L_tfm_normed), np.min(L_mm_cvxpy_normed)])
    Lmm_data = np.load(f'{ufm_verysmall_result}/Lmm_cvxpy_verysmallset.npy')
    Lmm_data_normed = Lmm_data / np.linalg.norm(Lmm_data)

    # generate and save separate plots
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, dpi=100)
    im1 = axs.imshow(L_ufm_normed.T, cmap='hot', interpolation='nearest')

    # axs.set_title('L_ufm')
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im1)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'{figure_dir}/L_ufm_verysmall.pdf')

    fig, axs = plt.subplots(nrows, ncols, dpi=100)
    im2 = axs.imshow(L_tfm_normed.T, cmap='hot', interpolation='nearest')
    # axs.set_title('L_tfm')
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im2)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'{figure_dir}/L_tfm_verysmall.pdf')

    fig, axs = plt.subplots(nrows, ncols, dpi=100)
    im3 = axs.imshow(Lmm_data_normed.T, cmap='hot', interpolation='nearest')
    # axs.set_title('L_mm_cvxpy')
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im3)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'{figure_dir}/L_mm_cvxpy_verysmall.pdf')
    # plt.show()

    # generate and save a plot with all 3 matrices on the same colorbar
    nrows, ncols = 1, 4
    fsize=22
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    im1 = axs[2].imshow(L_ufm.T, cmap='hot', interpolation='nearest')
    # axs[2].set_title('UFM Logits', fontsize=fsize)
    im2 = axs[3].imshow(L_tfm.T, cmap='hot', interpolation='nearest')
    # axs[3].set_title('TF logits', fontsize=fsize)
    im3 = axs[1].imshow(Lmm_data_normed.T, cmap='hot', interpolation='nearest')
    # axs[1].set_title('max margin solution L', fontsize=fsize)
    im4 = axs[0].imshow(S.T, cmap='hot', interpolation='nearest')
    # axs[0].set_title('Support Set',fontsize=fsize)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.colorbar(im1)
    # plt.colorbar(im2)
    # plt.colorbar(im3)
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    fig.tight_layout()
    plt.savefig(f'{figure_dir}/L_ufm_tfm_mm_verysmallset.pdf')

def plot_logits_tiny404():
    L_list_ufm = np.load(
        f'{ufm_tiny404_result}/L_list_UFM_d128.npy')
    L_list_tfm = np.load(
        f'{tfm_tiny404_result}/L_list_4 layer Transformer pos off_d128.npy')
    L_list_lstm = np.load(
        f'{lstm_tiny404_result}/L_list_LSTM_d128.npy')
    # rec = np.load(f'/Users/Lenovo/Next_Token/ExpPaperV1/results_tiny_extract_m404_cpu_ufm128_404_sori/rec_UFM_d128.npy')
    L_ufm = L_list_ufm[-1]
    L_ufm_normed = L_ufm / np.linalg.norm(L_ufm)
    L_tfm = L_list_tfm[-1]
    L_tfm_normed = L_tfm / np.linalg.norm(L_tfm)
    L_lstm = L_list_lstm[-1]
    L_lstm_normed = L_ufm / np.linalg.norm(L_lstm)
    # imshow L_ufm, L_tfm and L_mm_cvxpy on the same colorbar
    # color_max = np.max([np.max(L_ufm_normed), np.max(L_tfm_normed), np.max(L_mm_cvxpy_normed)])
    # color_min = np.min([np.min(L_ufm_normed), np.min(L_tfm_normed), np.min(L_mm_cvxpy_normed)])
    Lmm_data = np.load(f'{ufm_tiny404_result}/Lmm_cvxpy_tiny_extract_m404.npy')
    Lmm_data_normed = Lmm_data / np.linalg.norm(Lmm_data)

    # generate and save separate plots
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, dpi=100)
    im1 = axs.imshow(L_ufm_normed.T, cmap='hot', interpolation='nearest')

    # axs.set_title('L_ufm')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im1)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'{figure_dir}/L_ufm_tiny404.pdf')

    fig, axs = plt.subplots(nrows, ncols, dpi=100)
    im2 = axs.imshow(L_tfm_normed.T, cmap='hot', interpolation='nearest')
    # axs.set_title('L_tfm')
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im2)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'{figure_dir}/L_tfm_tiny404.pdf')

    fig, axs = plt.subplots(nrows, ncols, dpi=100)
    im3 = axs.imshow(L_lstm_normed.T, cmap='hot', interpolation='nearest')
    # axs.set_title('L_mm_cvxpy')
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im3)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'{figure_dir}/L_lstm_tiny404.pdf')

    fig, axs = plt.subplots(nrows, ncols, dpi=100)
    im4 = axs.imshow(Lmm_data_normed.T, cmap='hot', interpolation='nearest')
    # axs.set_title('L_mm_cvxpy')
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im4)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'{figure_dir}/L_mm_cvxpy_tiny404.pdf')
    # plt.show()

    # generate and save a plot with all 3 matrices on the same colorbar
    nrows, ncols = 1, 4
    fsize=22
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    im1 = axs[2].imshow(L_ufm.T, cmap='hot', interpolation='nearest')
    # axs[2].set_title('UFM Logits', fontsize=fsize)
    im2 = axs[3].imshow(L_tfm.T, cmap='hot', interpolation='nearest')
    # axs[3].set_title('TF logits', fontsize=fsize)
    im3 = axs[1].imshow(Lmm_data_normed.T, cmap='hot', interpolation='nearest')
    # axs[1].set_title('max margin solution L', fontsize=fsize)
    im4 = axs[0].imshow(S.T, cmap='hot', interpolation='nearest')
    # axs[0].set_title('Support Set',fontsize=fsize)
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.colorbar(im1)
    # plt.colorbar(im2)
    # plt.colorbar(im3)
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    fig.tight_layout()
    plt.savefig(f'{figure_dir}/L_ufm_tfm_mm_s_tiny404.pdf')




if check_lmm:
    # S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
    # L_mm_cvxpy = SVM_cvxpy_sol(S).T
    # L_mm_cvxpy_normed = L_mm_cvxpy / np.linalg.norm(L_mm_cvxpy)
    # L_mm_thm = SVM_thm_sol(tsk.ctx_dict, support_set_sampled, v_nt)
    # L_mm_thm_normed = L_mm_thm / np.linalg.norm(L_mm_thm)
    mean_diff = np.mean(np.abs(L_mm_cvxpy - L_mm_thm))
    # satisfy_Lmm_thm = check_lmm_constraint(S, L_mm_thm)
    # satisfy_Lmm_cvxpy = check_lmm_constraint(S, L_mm_cvxpy)
    # satisfy_Lmm_train = check_lmm_constraint(S, L_list[-1])
    print(f"nuclear norm of Lmm_thm: {np.linalg.norm(L_mm_thm, 'nuc')}")
    print(f"nuclear norm of Lmm_cvxpy: {np.linalg.norm(L_mm_cvxpy, 'nuc')}")
    print(f"nuclear norm of trained L: {np.linalg.norm(L_list[-1], 'nuc')}")
    print(f"cvxpy and thm L_mm diff for {dataset_name}: {np.linalg.norm(L_mm_cvxpy - L_mm_thm)}")
    print(f"cvxpy and thm L_mm normed diff for {dataset_name}: {np.linalg.norm(L_mm_cvxpy_normed - L_mm_thm_normed)}")
    print(f"np.mean(np.abs(L_mm_cvxpy - L_mm_thm)) : {mean_diff}")
    # print(f"Lmm_thm satisfy: {satisfy_Lmm_thm}")
    # print(f"Lmm_cvxpy satisfy: {satisfy_Lmm_cvxpy}")

    #plot 2 matrices with imshow on the same colorbar
    color_max = np.max([np.max(L_mm_cvxpy), np.max(L_mm_thm)])
    color_min = np.min([np.min(L_mm_cvxpy), np.min(L_mm_thm)])

    nrows, ncols = 1, 2
    fig, axs = plt.subplots(nrows, ncols, dpi=100)
    im1 = axs[0].imshow(L_mm_cvxpy.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)
    axs[0].set_title('L_mm_cvxpy')
    im2 = axs[1].imshow(L_mm_thm.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)
    axs[1].set_title('L_mm_thm')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.savefig(f'{figure_dir}/L_mm_cvxpy_thm_{model_name}_d{dim}.pdf')
    # plt.show()

    # exit(0)

HG_diff_cvxpy = np.zeros((len(W_list),1))
WG_diff_cvxpy = np.zeros((len(W_list),1))
HG_diff_thm = np.zeros((len(W_list),1))
sqrt_GH_diff_thm = np.zeros((len(W_list),1))
sqrt_GW_diff_thm = np.zeros((len(W_list),1))

WG_diff_thm = np.zeros((len(W_list),1))

HG_proj_diff_cvxpy = np.zeros((len(W_list),1))

for j in range(len(W_list)):
    ww = W_list[j]
    hh = H_list[j]

    WG = ww @ ww.T
    HG = hh @ hh.T

    GW_proj = project_W2mm(WG, vh_cvxpy)

    WG_norm = WG / norm(WG,'fro')
    HG_norm = HG / norm(HG,'fro')
    GW_proj_norm = GW_proj / norm(GW_proj, 'fro')

    wdiff_cvxpy = WG_norm - WG_mm_norm_cvxpy
    WG_diff_cvxpy[j] = norm(wdiff_cvxpy, 'fro')

    wproj_diff_cvxpy = GW_proj_norm - WG_mm_norm_cvxpy
    HG_proj_diff_cvxpy[j] = norm(wproj_diff_cvxpy, 'fro')

    wdiff_thm = WG_norm - WG_mm_norm_thm
    WG_diff_thm[j] = norm(wdiff_thm, 'fro')

    hdiff_cvxpy = HG_norm - HG_mm_norm_cvxpy
    HG_diff_cvxpy[j] = norm(hdiff_cvxpy, 'fro')

    hdiff_thm = HG_norm - HG_mm_norm_thm
    HG_diff_thm[j] = norm(hdiff_thm, 'fro')

    sqrt_GW_diff_thm[j] = norm(HG - sqrt_GL_mm_thm, 'fro')
    sqrt_GH_diff_thm[j] = norm(WG - sqrt_GL_mm_thm_T, 'fro')


def plot_norms():

    nrows, ncols = 2,3
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    axs[0,0].plot(rec, W_norm, label='||W_t||')
    axs[0,0].plot(rec, H_norm, label='||H_t||')
    axs[0,0].set_title('Norm')
    axs[0,0].set_xlabel(r'Epoch $k$')
    axs[0,0].legend()


    # if init_from == "ufm":
    # L_Lmm_diff = compute_norm_corr(L_list, l_mm_normed)
    L_Lmm_cvxpy_diff = compute_norm_corr(L_list, L_mm_cvxpy_normed)
    L_Lmm_thm_diff = compute_norm_corr(L_list, L_mm_thm_normed)

    axs[0, 1].plot(rec, L_Lmm_cvxpy_diff,label="||L - Lmm_cvxpy||", linestyle='--', color='green')
    axs[0, 1].plot(rec, L_Lmm_thm_diff, label="||L - Lmm_thm||")
    axs[0, 1].set_title(f'||L - Lmm||')
    axs[0, 1].set_xlabel('epoch')
    axs[0, 1].legend()

    axs[1, 0].plot(rec, WG_diff_thm, label="||GW_t - GW_mm|| theorem")
    axs[1, 0].plot(rec, WG_diff_cvxpy, label="||GW_t - GW_mm|| cvxpy", linestyle='--', color='green')
    # axs[1, 0].plot(rec, HG_proj_diff_cvxpy, label="||GW_t_proj - GW_mm|| cvxpy", linestyle='--', color='red')
    axs[1, 0].plot(rec, sqrt_GW_diff_thm, label="||GW_t - sqrt(GL_mmT) thm||", linestyle='--', color='red')
    axs[1, 0].set_title('GW_t Convergence')
    axs[1, 0].set_xlabel('epoch')
    axs[1, 0].legend()

    axs[1, 1].plot(rec, HG_diff_thm, label="||GH_t - GH_mm|| theorem")
    axs[1, 1].plot(rec, HG_diff_cvxpy, label="||GH_t - GH_mm|| cvxpy", linestyle='--', color='green')
    axs[1, 1].plot(rec, sqrt_GH_diff_thm, label="||GH_t - sqrt(GL_mm) thm||", linestyle='--', color='red')
    axs[1, 1].set_title('GH_t Convergence')
    axs[1, 1].set_xlabel('epoch')
    axs[1, 1].legend()

    axs[1, 2].plot(rec, L_proj_L_fin_diff, label="||L_proj - L_fin||")
    axs[1, 2].set_title('||L_proj - L_fin||')
    axs[1, 2].set_xlabel('epoch')
    axs[1, 2].legend()

    axs[0, 2].plot(rec, L_train_L_fin_Lmm_cvxpy, label="||(L-L_fin) - Lmm_cvxpy||", linestyle='--', color='green')
    axs[0, 2].plot(rec, L_train_L_fin_Lmm_thm, label="||(L-L_fin) - Lmm_thm||")
    axs[0, 2].plot(rec, L_Lmm_cvxpy_diff,label="||L - Lmm_cvxpy||", linestyle='--', color='red')
    axs[0, 2].plot(rec, L_Lmm_thm_diff, label="||L - Lmm_thm||")
    axs[0, 2].set_yscale('log')
    axs[0, 2].set_title('||(L-L_fin) - Lmm||')
    axs[0, 2].set_xlabel('epoch')
    axs[0, 2].legend()


    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    # Show the plot
    plt.savefig(f'{figure_dir}/Wnorm_{model_name}_d{dim}.pdf')
    # plt.show()


# plot_norms()
# plotlossverysmall()
# plotloss404()
# W_H_norm_plot()
# plot_logits_verysmallsets()
# plot_logits_tiny404()
# plot_logits_verysmallsets()

def check_convergence_plot():
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    axs[0, 0].plot(rec, W_norm, label='||W_t||')
    axs[0, 0].plot(rec, H_norm, label='||H_t||')
    axs[0, 0].set_title('Norm')
    axs[0, 0].set_xlabel('epoch')
    axs[0, 0].legend()


    axs[0, 1].plot(rec[1:], W_dir_norm)
    axs[0, 1].set_title('||W_dir||')
    axs[0, 1].set_xlabel('epoch')

    axs[1, 0].plot(rec[1:], L_dir_norm)
    axs[1, 0].set_title('||L_dir||')
    axs[1, 0].set_xlabel('epoch')

    axs[1, 1].plot(rec[1:], H_dir_norm)
    axs[1, 1].set_title('||H_dir||')
    axs[1, 1].set_xlabel('epoch')

    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    # Show the plot
    plt.savefig(f'{figure_dir}/Conv_{model_name}_d{dim}.pdf')
    # plt.show()

# check_convergence_plot()

def matrix_plot(WG_mm_norm, HG_mm_norm, l_mm_normed, lmm_name):
    last_GW = W_list_normed[-1] @ W_list_normed[-1].T
    last_GH = H_list_normed[-1] @ H_list_normed[-1].T
    last_WG_norm = last_GW / norm(last_GW, 'fro')
    last_HG_norm = last_GH / norm(last_GH, 'fro')


    nrows, ncols = 2, 3
    L_min = min(min(l_mm_normed.flatten()), min(L_list_normed[-1].flatten()))
    L_max = max(max(l_mm_normed.flatten()), max(L_list_normed[-1].flatten()))

    GW_min = min(min(last_WG_norm.flatten()), min(WG_mm_norm[-1].flatten()))
    GW_max = max(max(last_WG_norm.flatten()), max(WG_mm_norm[-1].flatten()))

    GH_min = min(min(last_HG_norm.flatten()), min(HG_mm_norm[-1].flatten()))
    GH_max = max(max(last_HG_norm.flatten()), max(HG_mm_norm[-1].flatten()))

    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    im1 = axs[0, 0].imshow(L_list_normed[-1].T, cmap='hot', interpolation='nearest',vmin = L_min, vmax = L_max)
    axs[0, 0].set_title('L(trained)')

    im2 = axs[1, 0].imshow(l_mm_normed.T, cmap='hot', interpolation='nearest',vmin = L_min, vmax = L_max)
    axs[1, 0].set_title(f'L_mm, {lmm_name}')

    im3 = axs[0, 1].imshow(last_WG_norm, cmap='hot', interpolation='nearest',vmin = GW_min, vmax = GW_max)
    axs[0, 1].set_title('GW(trained)')

    im4 = axs[1, 1].imshow(WG_mm_norm, cmap='hot', interpolation='nearest',vmin = GW_min, vmax = GW_max)
    axs[1, 1].set_title('GW_mm')

    im5 = axs[0, 2].imshow(last_HG_norm, cmap='hot', interpolation='nearest',vmin = GH_min, vmax = GH_max)
    axs[0, 2].set_title('GH(trained)')

    im6 = axs[1, 2].imshow(HG_mm_norm, cmap='hot', interpolation='nearest',vmin = GH_min, vmax = GH_max)
    axs[1, 2].set_title('GH_mm')

    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    for im_ in [im1, im2, im3, im4, im5, im6]:
        plt.colorbar(im_)
    plt.savefig(f'{figure_dir}/Matrices_{model_name}_d{dim}_{lmm_name}.pdf')
    # plt.show()

def matrix_plot_sqrt():

    # S, P H, W, L, H_mm, W_mm, L_mm
    # S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
    # P = get_p(tsk.ctx_dict, support_set_sampled, support_set_pr, v_nt )
    H = H_list[-1]
    W = W_list[-1]
    L = L_list[-1]

    su, ss, svh = np.linalg.svd(S)
    ss_aqrt = np.sqrt(ss)

    sigmau = np.zeros(su.shape)
    np.fill_diagonal(sigmau, ss_aqrt)

    U_sigma = su @ sigmau

    sigmavh = np.zeros(svh.shape)
    np.fill_diagonal(sigmavh, ss_aqrt)
    Vh_sigma = svh.T @ sigmavh

    plot_matrices = [S, H, S.T, W]
    plot_matrices_names = ["STS", "HTH" , "SST", "WWT"]
    fill_list = [[0,0], [0,1], [1,0], [1,1]]




    nrows, ncols = 2, 5
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    i = 0
    remove = []
    for r, c in fill_list:
        A = plot_matrices[i]
        N = plot_matrices_names[i]

        AAT = A@A.T
        # AAT = AAT/norm(AAT)

        # AAT_min, AAT_max = -0.03, 0.25

        im1 = axs[r, c].imshow(AAT, cmap='viridis', interpolation='nearest')
        axs[r, c].set_title(N)

        cb = plt.colorbar(im1)

        i=i+1

    # plt.savefig(f'{figure_dir}/Matrices_sqrt_{model_name}_{dataset_name}.pdf')
    # svd the SST
    su, ss, svh = np.linalg.svd(S@S.T)
    sigma = np.zeros((su.shape[1], svh.shape[0]))
    np.fill_diagonal(sigma, np.sqrt(ss))
    sqrt_ssT = su @ sigma @ svh
    # sqrt_ssT = sqrt_ssT/norm(sqrt_ssT)
    # sqrt_ssT_min, sqrt_ssT_max = min(sqrt_ssT_norm.flatten()), max(sqrt_ssT_norm.flatten())
    im1 = axs[0, 4].imshow(sqrt_ssT, cmap='viridis', interpolation='nearest')
    axs[0, 4].set_title('sqrt(STS)')
    plt.colorbar(im1)

    #sqrt(sTs)
    su, ss, svh = np.linalg.svd(S.T@S)
    sigma = np.zeros((su.shape[1], svh.shape[0]))
    np.fill_diagonal(sigma, np.sqrt(ss))
    sqrt_ssT = su @ sigma @ svh
    # sqrt_ssT = sqrt_ssT/norm(sqrt_ssT)
    # sqrt_sTs_min, sqrt_sTs_max = min(sqrt_sTs_norm.flatten()), max(sqrt_sTs_norm.flatten())
    im2 = axs[1, 4].imshow(sqrt_ssT, cmap='viridis', interpolation='nearest')
    axs[1, 4].set_title('sqrt(SST)')
    plt.colorbar(im2)

    # plot HG_mm_cvxpy
    im3 = axs[0, 2].imshow(HG_mm_cvxpy, cmap='viridis', interpolation='nearest')
    axs[0, 2].set_title('H_mmTHmm(cvxpy)')
    plt.colorbar(im3)


    # plot WG_mm_cvxpy
    im4 = axs[1, 2].imshow(WG_mm_cvxpy, cmap='viridis', interpolation='nearest')
    axs[1, 2].set_title('W_mmTWmm(cvxpy)')
    plt.colorbar(im4)

    # plot HG_mm_thm
    im5 = axs[0, 3].imshow(HG_mm_thm, cmap='viridis', interpolation='nearest')
    axs[0, 3].set_title('H_mmTHmm(conjectured)')
    plt.colorbar(im5)

    # plot WG_mm_thm
    im6 = axs[1, 3].imshow(WG_mm_thm, cmap='viridis', interpolation='nearest')
    axs[1, 3].set_title('W_mmTWmm(conjectured)')
    plt.colorbar(im6)



    plt.savefig(f'{figure_dir}/Matrices_sqrt_unormed_mm_{model_name}_{dataset_name}.pdf')
    # plt.show()
def matrix_sim_plot_sqrt():

    # S, P H, W, L, H_mm, W_mm, L_mm
    # S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
    # P = get_p(tsk.ctx_dict, support_set_sampled, support_set_pr, v_nt )
    H = H_list[-1]
    W = W_list[-1]
    L = L_list[-1]

    su, ss, svh = np.linalg.svd(S)
    ss_aqrt = np.sqrt(ss)

    sigmau = np.zeros(su.shape)
    np.fill_diagonal(sigmau, ss_aqrt)

    U_sigma = su @ sigmau

    sigmavh = np.zeros(svh.shape)
    np.fill_diagonal(sigmavh, ss_aqrt)
    Vh_sigma = svh.T @ sigmavh

    plot_matrices = [S, H, U_sigma, S.T, W, Vh_sigma]
    plot_matrices_names = ["Cos(S,S)", "Cos(H,H)", "Cos(sqrt(S),sqrt(S))" , "Cos(ST, ST)", "Cos(W, W)", "Cos(sqrt(ST),sqrt(ST))"]




    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    i = 0
    for r in range(nrows):
        for c in range(ncols):
            A = plot_matrices[i]
            N = plot_matrices_names[i]

            # AAT = A@A.T
            # AAT_norm = AAT/norm(AAT)
            Asim = pairwise_cossim(A)

            AAT_min, AAT_max = -1, 1

            im1 = axs[r, c].imshow(Asim, cmap='viridis', interpolation='nearest', vmin=AAT_min, vmax=AAT_max)
            axs[r, c].set_title(N)

            plt.colorbar(im1)
            i=i+1




    plt.savefig(f'{figure_dir}/Matrices_sim_sqrt_{model_name}_{dataset_name}.pdf')
    # plt.show()

# matrix_plot_sqrt()
# matrix_sim_plot_sqrt()
# exit()
def matrix_plot_all(H_mm, W_mm, L_mm, lmm_name):

    # S, P H, W, L, H_mm, W_mm, L_mm
    # S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
    # P = get_p(tsk.ctx_dict, support_set_sampled, support_set_pr, v_nt )
    H = H_list[-1]
    W = W_list[-1]
    L = L_list[-1]

    plot_matrices = [S, P, H_mm, W_mm, L_mm, H, W, L]
    plot_matrices_names = ["S", "P", "H_mm", "W_mm", f"L_mm {lmm_name}", "H", "W", "L"]

    nrows, ncols = 2, 8
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    i = 0
    ims = []
    AAT_min, AAT_max = 0.07, -0.02
    ATA_min, ATA_max = 0.07, -0.02
    for c in range(ncols):
        A = plot_matrices[i]
        N = plot_matrices_names[i]
        AAT = A@A.T
        AAT_norm = AAT/norm(AAT)
        ATA = A.T@A
        ATA_norm = ATA/norm(ATA)

        # AAT_min, AAT_max = min(AAT_norm.flatten()), max(AAT_norm.flatten())
        # ATA_min, ATA_max = min(ATA_norm.flatten()), max(ATA_norm.flatten())

        im1 = axs[0, c].imshow(AAT_norm, cmap='viridis', interpolation='nearest', vmin=AAT_min, vmax=AAT_max)
        axs[0, c].set_title(f'{N}{N}.T')
        im2 = axs[1, c].imshow(ATA_norm, cmap='viridis', interpolation='nearest', vmin=ATA_min, vmax=ATA_max)
        axs[1, c].set_title(f'{N}.T{N}')
        plt.colorbar(im1)
        plt.colorbar(im2)

        i+=1

    plt.savefig(f'{figure_dir}/Matrices_all_{model_name}_d{dim}_{lmm_name}.pdf')
    # plt.show()

def matrix_plot_all_nonormalization(H_mm, W_mm, L_mm, lmm_name):

    # S, P H, W, L, H_mm, W_mm, L_mm
    # S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
    # P = get_p(tsk.ctx_dict, support_set_sampled, support_set_pr, v_nt )
    H = H_list[-1]
    W = W_list[-1]
    L = L_list[-1]

    plot_matrices = [S, P, H_mm, W_mm, L_mm, H, W, L]
    plot_matrices_names = ["S", "P", "H_mm", "W_mm", f"L_mm {lmm_name}", "H", "W", "L"]

    nrows, ncols = 2, 8
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    i = 0
    ims = []

    for c in range(ncols):
        A = plot_matrices[i]
        N = plot_matrices_names[i]
        AAT = A@A.T
        AAT_norm = AAT
        ATA = A.T@A
        ATA_norm = ATA

        if c == 4:
            AAT_norm = matrix_sqrt(AAT)
            ATA_norm = matrix_sqrt(ATA)

        # AAT_min, AAT_max = min(AAT_norm.flatten()), max(AAT_norm.flatten())
        # ATA_min, ATA_max = min(ATA_norm.flatten()), max(ATA_norm.flatten())

        im1 = axs[0, c].imshow(AAT_norm, cmap='viridis', interpolation='nearest')
        axs[0, c].set_title(f'{N}{N}.T')
        im2 = axs[1, c].imshow(ATA_norm, cmap='viridis', interpolation='nearest')
        axs[1, c].set_title(f'{N}.T{N}')
        if c == 4:
            axs[0, c].set_title(f'sqrt {N}{N}.T')
            axs[0, c].set_title(f'sqrt {N}.T{N}')
        plt.colorbar(im1)
        plt.colorbar(im2)

        i+=1

    plt.savefig(f'{figure_dir}/Matrices_all_nonorm_{model_name}_{dataset_name}_{lmm_name}.pdf')
    # plt.show()

# matrix_plot_all_nonormalization(H_mm_thm, W_mm_thm, L_mm_thm, 'thm')
def Sim_plot_all(H_mm, W_mm, L_mm,  lmm_name):

    # S, P H, W, L, H_mm, W_mm, L_mm
    # S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
    # P = get_p(tsk.ctx_dict, support_set_sampled, support_set_pr, v_nt )
    H = H_list[-1]
    W = W_list[-1]
    L = L_list[-1]

    plot_matrices = [S, P, H_mm, W_mm, L_mm, H, W, L]
    plot_matrices_names = ["S", "P", "H_mm", "W_mm", f"L_mm {lmm_name}", "H", "W", "L"]

    nrows, ncols = 2, 8
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    AAT_min, AAT_max = 1, -1
    ATA_min, ATA_max = 1, -1

    i = 0
    for c in range(ncols):
        A = plot_matrices[i]
        N = plot_matrices_names[i]
        # AAT = A@A.T
        # AAT_norm = AAT/norm(AAT)
        Asim = pairwise_cossim(A)
        # ATA = A.T@A
        # ATA_norm = ATA/norm(ATA)
        ATsim = pairwise_cossim(A.T)

        # AAT_min, AAT_max = min(Asim.flatten()), max(Asim.flatten())
        # ATA_min, ATA_max = min(ATsim.flatten()), max(ATsim.flatten())

        im1 = axs[0, c].imshow(Asim, cmap='viridis', interpolation='nearest', vmin=AAT_min, vmax=AAT_max)
        axs[0, c].set_title(f'{N} row cos similarity')
        im2 = axs[1, c].imshow(ATsim, cmap='viridis', interpolation='nearest', vmin=ATA_min, vmax=ATA_max)
        axs[1, c].set_title(f'{N} col cos similarity')
        plt.colorbar(im1)
        plt.colorbar(im2)

        i += 1

    plt.savefig(f'{figure_dir}/cossim_all_{model_name}_d{dim}_{lmm_name}.pdf')


def Sim_plot_iter(H_mm, W_mm, L_mm,  lmm_name):

    # S, P H, W, L, H_mm, W_mm, L_mm
    # S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
    # P = get_p(tsk.ctx_dict, support_set_sampled, support_set_pr, v_nt )
    it_list = [0,2,3]
    label_list = [0, 10, 100, 3000, -1]
    H_1000 = np.load(f'{result_dir}/H_list_{model_name}_d{dim}_1000.npy')
    W_1000 = np.load(f'{result_dir}/W_list_{model_name}_d{dim}_1000.npy')
    L_1000 = np.load(f'{result_dir}/L_list_{model_name}_d{dim}_1000.npy')


    H_list_plot = [H_1000[i] for i in it_list]
    H_list_plot.append(H_list[-5])
    W_list_plot = [W_1000[i] for i in it_list]
    W_list_plot.append(W_list[-5])
    L_list_plot = [L_1000[i] for i in it_list]
    L_list_plot.append([L_list[-5]])


    # H = H_list[-1]
    # W = W_list[-1]
    # L = L_list[-1]

    # plot_matrices = [S, P, H_mm, W_mm, L_mm, H, W, L]
    # plot_matrices_names = ["S", "P", "H_mm", "W_mm", f"L_mm {lmm_name}", "H", "W", "L"]

    nrows, ncols = 3, len(label_list)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=100)

    AAT_min, AAT_max = 1, -1
    ATA_min, ATA_max = 1, -1

    for i,it in enumerate(label_list):
        print(i)
        if it == 0:
            Nh = f'H Initialization'
            H = H_list_init
            Nw = f'W Initialization'
            W = W_list_init
            Nl = f'L Initialization'
            L = L_list_init


        elif it == -1:
            Nh = f'H_mm'
            H = H_mm
            Nw = f'W_mm'
            W = W_mm
            Nl = f'L_mm'
            L = L_mm

        else:
            H = H_list_plot[i]
            Nh = f'H step {label_list[i]}'
            W = W_list_plot[i]
            Nw = f'W step {label_list[i]}'
            L = L_list_plot[i]
            Nl = f'L step {label_list[i]}'

        print(H.shape)
        Hsim = pairwise_cossim(H)
        Wsim = pairwise_cossim(W)

        # AAT_min, AAT_max = min(Asim.flatten()), max(Asim.flatten())
        # ATA_min, ATA_max = min(ATsim.flatten()), max(ATsim.flatten())

        im1 = axs[0, i].imshow(Hsim, cmap='viridis', interpolation='nearest', vmin=AAT_min, vmax=AAT_max)
        axs[0, i].set_title(f'{Nh} cos similarity')
        im2 = axs[1, i].imshow(Wsim, cmap='viridis', interpolation='nearest', vmin=ATA_min, vmax=ATA_max)
        axs[1, i].set_title(f'{Nw} cos similarity')
        im3 = axs[2, i].imshow(np.array(L).T, cmap='hot', interpolation='nearest')
        axs[2, i].set_title(f'{Nl}')
        plt.colorbar(im1)
        plt.colorbar(im2)
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.savefig(f'{figure_dir}/cossim_steps{dataset_name}_{model_name}_d{dim}_{lmm_name}.pdf')
#
# Sim_plot_iter(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, 'cvxpy')
# exit(0)

def Sim_plot_paper(H_mm, W_mm):

    # S, P H, W, L, H_mm, W_mm, L_mm
    # S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
    # P = get_p(tsk.ctx_dict, support_set_sampled, support_set_pr, v_nt )
    H = H_list[-1]
    W = W_list[-1]
    L = L_list[-1]

    plot_matrices = [S, H_mm, H, S.T, W_mm, W]

    plot_matrices_names = ["S", "H_mm", "H", "S_T", "W_mm", "W"]



    i = 0
    for i in range(len(plot_matrices)):
        nrows, ncols = 1,1
        fig, axs = plt.subplots(nrows, ncols, dpi=100)

        AAT_min, AAT_max = 1, -1

        A = plot_matrices[i]
        N = plot_matrices_names[i]

        Asim = pairwise_cossim(A)

        im1 = axs.imshow(Asim, cmap='viridis', interpolation='nearest', vmin=AAT_min, vmax=AAT_max)
        # axs.set_title(f'{N} row cos similarity')
        plt.colorbar(im1)

        plt.savefig(f'{figure_dir}/cossim_all_{model_name}_{dataset_name}_{N}.pdf')


# Sim_plot_paper(H_mm_cvxpy, W_mm_cvxpy)

def embedding_similarity_plot(H_mm, lmm_name):
    G_support = get_support_gram(tsk.ctx_dict, support_set_sampled, v_nt)
    # print(L_list_normed[-1])
    G_L_last = L_list_normed[-1] @ L_list_normed[-1].T
    sim_L_last = pairwise_cossim(L_list_normed[-1])
    # print(G_L_last)
    G_H_last = H_list_normed[-1] @ H_list_normed[-1].T
    sim_H_last = pairwise_cossim(H_list_normed[-1])

    Sim_H_mm = pairwise_cossim(H_mm)

    G_S_min, G_S_max = min(G_support.flatten()), max(G_support.flatten())
    G_Ll_min, G_Ll_max = min(G_L_last.flatten()), max(G_L_last.flatten())
    G_Hl_min, G_Hl_max = min(G_H_last.flatten()), max(G_H_last.flatten())

    sim_Ll_min, sim_Ll_max = min(sim_L_last.flatten()), max(sim_L_last.flatten())
    sim_Hl_min, sim_Hl_max = min(sim_H_last.flatten()), max(sim_H_last.flatten())

    sim_Hmm_min, sim_Hmm_max = min(Sim_H_mm.flatten()), max(Sim_H_mm.flatten())

    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    im1 = axs[0,0].imshow(G_L_last, cmap='summer', interpolation='nearest', vmin=G_Ll_min, vmax=G_Ll_max)
    axs[0,0].set_title('Trained Logits Gram')

    im2 = axs[1,2].imshow(G_support, cmap='summer', interpolation='nearest', vmin=G_S_min, vmax=G_S_max)
    axs[1,2].set_title('support Gram')

    im3 = axs[0,1].imshow(G_H_last, cmap='summer', interpolation='nearest', vmin=G_Hl_min, vmax=G_Ll_max)
    axs[0,1].set_title('Trained Embedding Gram')

    im4 = axs[1,0].imshow(sim_L_last, cmap='summer', interpolation='nearest', vmin=sim_Ll_min, vmax=sim_Ll_max)
    axs[1,0].set_title('Trained Logits pairwise similarity')

    im5 = axs[1,1].imshow(sim_H_last, cmap='summer', interpolation='nearest', vmin=sim_Hl_min, vmax=sim_Hl_max)
    axs[1,1].set_title('Trained Embedding pairwise similarity')

    im6 = axs[0,2].imshow(Sim_H_mm, cmap='summer', interpolation='nearest', vmin=sim_Hmm_min, vmax=sim_Hmm_max)
    axs[0,2].set_title(f'Hmm Pairwise_similarity, {lmm_name}')

    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    for im_ in [im1, im2, im3, im4, im5]:
        plt.colorbar(im_)
    plt.savefig(f'{figure_dir}/Suppot_sim_{model_name}_d{dim}_{lmm_name}.pdf')
    # plt.show()

# def redo_figure3_verysmall(H_mm, W_mm, L_mm,  lmm_name):
#     UFMH = H_list_ufmsmall[-1]
#     UFMW = W_list_ufmsmall[-1]
#     UFML = L_list_ufmsmall[-1]
#
#     TFH = H_list_tfsmall[-1]
#     TFW = W_list_tfsmall[-1]
#     TFL = L_list_tfsmall[-1]
#
#     Lmm_data = np.load(f'{tfm_verysmall_result}/Lmm_cvxpy_verysmallset.npy')
#     Lmm_data_normed = Lmm_data / np.linalg.norm(Lmm_data)
#
#     plot_matrices = [[S, H_mm, UFMH, TFH],[ S.T, W_mm, UFMW, TFW], [S, L_mm, UFML, TFL]]
#
#     # plot_matrices =
#     plot_matrices_names = [["S", "H_mm", "UFMH", "TFH"],\
#                            ["ST", "W_mm", "UFMW", "TFW"],\
#                             ["S", "L_mm", "UFML", "TFL"]]
#
#     plot_title_names = [[r"$COS(S,S)$", r"$COS(H^{mm},H^{mm})$", r"$UFM: COS(H,H)$", r"$TF: COS(H,H)$"],\
#                         [r"$COS(S^T,S^T)$", r"$COS(W^{mm^T},W^{mm^T}$", r"$UFM: COS(W^T,W^T)$", r"$TF: COS(W^T,W^T)$"],\
#                         [r"$S$", r"$L^{mm}$", r"UFM: L", r"TF: L"]]
#
#     nrows, ncols = 3, 4
#     fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
#
#     AAT_min, AAT_max = 1, -1
#     ATA_min, ATA_max = 1, -1
#
#     i = 0
#     reodering = [3,2,1,0]
#     for r in range(2):
#         for c_ in range(ncols):
#             c = reodering[c_]
#             # print(r,c)
#             A = plot_matrices[r][c]
#             # N = plot_matrices_names[r][i]
#             # AAT = A@A.T
#             # AAT_norm
#             Asim = pairwise_cossim(A)
#             # ATA = A.T@A
#             # ATA_norm = ATA/norm(ATA)
#             # ATsim = pairwise_cossim(A.T)
#
#         # AAT_min, AAT_max = min(Asim.flatten()), max(Asim.flatten())
#         # ATA_min, ATA_max = min(ATsim.flatten()), max(ATsim.flatten())
#             im1 = axs[r, c_].imshow(Asim, cmap='viridis', interpolation='nearest', vmin=AAT_min, vmax=AAT_max)
#             # axs[r, c_].set_title(f'{plot_title_names[r][c]}')
#             plt.colorbar(im1)
#
#     im1 = axs[2, reodering[2]].imshow(UFML.T, cmap='hot', interpolation='nearest')
#     # axs[2, 2].set_title(f'{plot_title_names[2][reodering[2]]}')
#     im2 = axs[2, reodering[3]].imshow(TFL.T, cmap='hot', interpolation='nearest')
#     # axs[2, 3].set_title(f'{plot_title_names[2][reodering[3]]}')
#     im3 = axs[2, reodering[1]].imshow(Lmm_data_normed.T, cmap='hot', interpolation='nearest')
#     # axs[2, 1].set_title(f'{plot_title_names[2][reodering[1]]}')
#     im4 = axs[2, reodering[0]].imshow(S.T, cmap='hot', interpolation='nearest')
#     # axs[2, 0].set_title(f'{plot_title_names[2][reodering[0]]}')
#
#     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.3)
#
#     plt.savefig(f'{figure_dir}/figure3_verysmall_{model_name}_d{dim}_{lmm_name}.pdf')
#
# # redo_figure3_verysmall(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, 'cvxpy')
# # exit(0)
# def redo_figure3(H_mm, W_mm, L_mm,  lmm_name):
#     UFMH = H_list_ufm404[-1]
#     UFMW = W_list_ufm404[-1]
#     UFML = L_list_ufm404[-1]
#
#     TFH = H_list_tf404[-1]
#     TFW = W_list_tf404[-1]
#     TFL = L_list_tf404[-1]
#
#     Lmm_data = np.load(f'{ufm_tiny404_result}/Lmm_cvxpy_tiny_extract_m404.npy')
#     Lmm_data_normed = Lmm_data / np.linalg.norm(Lmm_data)
#
#     plot_matrices = [[S, H_mm, UFMH, TFH],[ S.T, W_mm, UFMW, TFW], [S, L_mm, UFML, TFL]]
#
#     # plot_matrices =
#     plot_matrices_names = [["S", "H_mm", "UFMH", "TFH"],\
#                            ["ST", "W_mm", "UFMW", "TFW"],\
#                             ["S", "L_mm", "UFML", "TFL"]]
#
#     plot_title_names = [[r"$COS(S,S)$", r"$COS(H^{mm},H^{mm})$", r"$UFM: COS(H,H)$", r"$TF: COS(H,H)$"],\
#                         [r"$COS(S^T,S^T)$", r"$COS(W^{mm^T},W^{mm^T}$", r"$UFM: COS(W^T,W^T)$", r"$TF: COS(W^T,W^T)$"],\
#                         [r"$S$", r"$L^{mm}$", r"UFM: L", r"TF: L"]]
#
#     nrows, ncols = 3, 4
#     fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
#
#     AAT_min, AAT_max = 1, -1
#     ATA_min, ATA_max = 1, -1
#
#     i = 0
#     reodering = [3,2,1,0]
#     for r in range(2):
#         for c_ in range(ncols):
#             c = reodering[c_]
#             # print(r,c)
#             A = plot_matrices[r][c]
#             # N = plot_matrices_names[r][i]
#             # AAT = A@A.T
#             # AAT_norm = AAT/norm(AAT)
#             Asim = pairwise_cossim(A)
#             # ATA = A.T@A
#             # ATA_norm = ATA/norm(ATA)
#             # ATsim = pairwise_cossim(A.T)
#
#         # AAT_min, AAT_max = min(Asim.flatten()), max(Asim.flatten())
#         # ATA_min, ATA_max = min(ATsim.flatten()), max(ATsim.flatten())
#
#             im1 = axs[r, c_].imshow(Asim, cmap='viridis', interpolation='nearest', vmin=AAT_min, vmax=AAT_max)
#             # axs[r, c_].set_title(f'{plot_title_names[r][c]}')
#             plt.colorbar(im1)
#
#
#     im1 = axs[2, reodering[2]].imshow(UFML.T, cmap='hot', interpolation='nearest')
#     # axs[2, 2].set_title(f'{plot_title_names[2][reodering[2]]}')
#     im2 = axs[2, reodering[3]].imshow(TFL.T, cmap='hot', interpolation='nearest')
#     # axs[2, 3].set_title(f'{plot_title_names[2][reodering[3]]}')
#     im3 = axs[2, reodering[1]].imshow(Lmm_data_normed.T, cmap='hot', interpolation='nearest')
#     # axs[2, 1].set_title(f'{plot_title_names[2][reodering[1]]}')
#     im4 = axs[2, reodering[0]].imshow(S.T, cmap='hot', interpolation='nearest')
#     # axs[2, 0].set_title(f'{plot_title_names[2][reodering[0]]}')
#
#     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.3)
#
#     plt.savefig(f'{figure_dir}/figure3_{model_name}_d{dim}_{lmm_name}.pdf')
#
#
# def redo_figure3(H_mm, W_mm, L_mm,  lmm_name):
#
#     UFMH = H_list_ufm404[-1]
#     UFMW = W_list_ufm404[-1]
#     UFML = L_list_ufm404[-1]
#
#     TFH = H_list_tf404[-1]
#     TFW = W_list_tf404[-1]
#     TFL = L_list_tf404[-1]
#     # Assuming the data arrays like `S` and other parameters are defined globally or passed to this function
#     # plot_matrices = [[S, H_mm, UFMH, TFH], [S.T, W_mm, UFMW, TFW], [S, L_mm, UFML, TFL]]
#     plot_matrices = [[TFH, UFMH, H_mm, H_mm_thm], [TFW, UFMW, W_mm, W_mm_thm], [TFL, UFML, L_mm, S]]
#     nrows, ncols = 3, 4
#     fig = plt.figure(figsize=(6 * ncols, 4 * nrows), dpi=100)
#
#     # Create GridSpec for the top two rows
#     gs_top = gridspec.GridSpec(2, 4, figure=fig,hspace=0.3, bottom=0.4, top=0.95)
#     gs_bottom = gridspec.GridSpec(1, 4, figure=fig, hspace=0.1, bottom=0.05, top=0.35)
#
#     # Plot the first two rows using the standard grid spec
#     for r in range(2):
#         for c in range(4):
#             ax = fig.add_subplot(gs_top[r, c])
#             Asim = pairwise_cossim(plot_matrices[r][c])
#             im = ax.imshow(Asim, cmap='viridis', interpolation='nearest', vmin=1, vmax=-1)
#             plt.colorbar(im, ax=ax)
#
#     # Plot the third row with a different GridSpec (gs_bottom)
#     for c in range(4):
#         ax = fig.add_subplot(gs_bottom[0, c])
#         Asim = (plot_matrices[2][c])
#         im = ax.imshow(Asim.T, cmap='hot', interpolation='nearest')
#         plt.colorbar(im, ax=ax)
#
#     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4)  # Adjust horizontal spacing in the bottom row
#     plt.savefig(f'{figure_dir}/figure3_{model_name}_d{dim}_{lmm_name}.pdf')
#     plt.show()


def redo_figure3(H_mm, W_mm, L_mm,  lmm_name):

    UFMH = H_list_ufm404[-1]
    UFMW = W_list_ufm404[-1]
    UFML = L_list_ufm404[-1]

    TFH = H_list_tf404[-1]
    TFW = W_list_tf404[-1]
    TFL = L_list_tf404[-1]
    # Assuming the data arrays like `S` and other parameters are defined globally or passed to this function
    # plot_matrices = [[S, H_mm, UFMH, TFH], [S.T, W_mm, UFMW, TFW], [S, L_mm, UFML, TFL]]
    plot_matrices = [
        [TFH, UFMH, H_mm, H_mm_thm],
        [TFW, UFMW, W_mm, W_mm_thm],
        [TFL, UFML, L_mm, L_mm_thm]
    ]
    nrows, ncols = 3, 4
    print(min(L_mm_thm.flatten()), max(L_mm_thm.flatten()))
    print(min(TFL.flatten()), max(TFL.flatten()))
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows), dpi=100)
    # gs = gridspec.GridSpec(nrows, ncols+1, figure=fig, width_ratios=[1, 1, 1, 1, 0.05])
    gs1 = gridspec.GridSpec(2, ncols+1, figure=fig, top=0.85, bottom=0.35, wspace=0.01, hspace=0.4, width_ratios=[1, 1, 1, 1, 0.05],left=0.05, right=0.9)
    gs2 = gridspec.GridSpec(1, ncols+1, figure=fig, top=0.3, bottom=0.1, wspace=0.4, hspace=0.4, width_ratios=[1, 1, 1, 1, 0.05], left=0.07, right=0.92)
    # Last column for color bars

    images = []
    for r in range(nrows):
        for c in range(ncols):

            if r < 2:
                ax = fig.add_subplot(gs1[r, c])
                Asim = pairwise_cossim(plot_matrices[r][c])
                im = ax.imshow(Asim, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
            else:
                if c < 2:
                    ax = fig.add_subplot(gs2[0, c])
                    Asim = plot_matrices[2][c].T
                    im = ax.imshow(Asim, cmap='hot', interpolation='nearest', vmin=-5, vmax=15)
                else:
                    ax = fig.add_subplot(gs2[0, c])
                    Asim = plot_matrices[2][c].T
                    im = ax.imshow(Asim, cmap='hot', interpolation='nearest', vmin=-0.5, vmax=1)
            images.append(im)

    # Create color bars in the last column for each row
    for r in range(nrows):
        if r < 2:
            cbar_ax = fig.add_subplot(gs1[r, -1])
        else:
            # get the position of last color bar
            # set position of the color bar to be the same as the last color bar
            cbar_ax = fig.add_subplot(gs2[0, -1])
            original_position = cbar_ax.get_position()
            # move left by 0.05
            cbar_ax.set_position([original_position.x0 - 0.02, original_position.y0, original_position.width, original_position.height])
            # cbar_ax = fig.add_subplot(gs2[0, -1])
        fig.colorbar(images[r * ncols+3], cax=cbar_ax)  # Use the first image of each row for the color bar

    # plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, wspace=0.1, hspace=0.4)  # Adjust margins and spacings
    plt.savefig(f'{figure_dir}/figure3_{model_name}_d{dim}_{lmm_name}.pdf')
    # plt.show()


# redo_figure3(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, 'cvxpy')
# exit(0)

def redo_figure3_verysmall(H_mm, W_mm, L_mm,  lmm_name):

    UFMH = H_list_ufmsmall[-1]
    UFMW = W_list_ufmsmall[-1]
    UFML = L_list_ufmsmall[-1]

    TFH = H_list_tfsmall[-1]
    TFW = W_list_tfsmall[-1]
    TFL = L_list_tfsmall[-1]
    # Assuming the data arrays like `S` and other parameters are defined globally or passed to this function
    # plot_matrices = [[S, H_mm, UFMH, TFH], [S.T, W_mm, UFMW, TFW], [S, L_mm, UFML, TFL]]
    plot_matrices = [
        [TFH, UFMH, H_mm, H_mm_thm],
        [TFW, UFMW, W_mm, W_mm_thm],
        [TFL, UFML, L_mm, L_mm_thm]
    ]
    nrows, ncols = 3, 4
    print(L_mm_thm)

    fig = plt.figure(figsize=(3 * ncols, 3 * nrows), dpi=100)
    # gs = gridspec.GridSpec(nrows, ncols+1, figure=fig, width_ratios=[1, 1, 1, 1, 0.05])
    gs1 = gridspec.GridSpec(2, ncols+1, figure=fig, top=0.85, bottom=0.35, wspace=0.01, hspace=0.4, width_ratios=[1, 1, 1, 1, 0.05],left=0.05, right=0.9)
    gs2 = gridspec.GridSpec(1, ncols+1, figure=fig, top=0.25, bottom=0.1, wspace=0.4, hspace=0.4, width_ratios=[1, 1, 1, 1, 0.05], left=0.07, right=0.92)
    # Last column for color bars

    images = []
    for r in range(nrows):
        for c in range(ncols):

            if r < 2:
                ax = fig.add_subplot(gs1[r, c])
                Asim = pairwise_cossim(plot_matrices[r][c])
                im = ax.imshow(Asim, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
            else:
                if c < 2:
                    ax = fig.add_subplot(gs2[0, c])
                    Asim = plot_matrices[2][c].T
                    im = ax.imshow(Asim, cmap='hot', interpolation='nearest', vmin=-5, vmax=15)
                else:
                    ax = fig.add_subplot(gs2[0, c])
                    Asim = plot_matrices[2][c].T
                    im = ax.imshow(Asim, cmap='hot', interpolation='nearest', vmin=-0.5, vmax=1)
            images.append(im)

    # Create color bars in the last column for each row
    for r in range(nrows):
        if r < 2:
            cbar_ax = fig.add_subplot(gs1[r, -1])
        else:
            # get the position of last color bar
            # set position of the color bar to be the same as the last color bar
            cbar_ax = fig.add_subplot(gs2[0, -1])
            original_position = cbar_ax.get_position()
            # move left by 0.05
            cbar_ax.set_position([original_position.x0 - 0.02, original_position.y0, original_position.width, original_position.height])
            # cbar_ax = fig.add_subplot(gs2[0, -1])
        fig.colorbar(images[r * ncols+3], cax=cbar_ax)  # Use the first image of each row for the color bar

    # plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, wspace=0.1, hspace=0.4)  # Adjust margins and spacings
    plt.savefig(f'{figure_dir}/figure3_verysmall_{model_name}_d{dim}_{lmm_name}.pdf')
    # plt.show()


# redo_figure3_verysmall(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, 'cvxpy')
# exit(0)

def W_sim_plot_with_label():
    W = W_list[-1]
    labels = []

    for z in range(v_nt):
        lbl = tsk.v_nt2v[z]
        lbl_str = sp.IdToPiece(int(lbl))
        print(z, lbl_str)
        labels.append(lbl_str)

    sim_W = pairwise_cossim(W)
    sim_W_min, sim_W_max = min(sim_W.flatten()), max(sim_W.flatten())
    nrows, ncols = 1, 1
    ftsz = 12
    fig, axs = plt.subplots(nrows, ncols, figsize=(30 * ncols, 25 * nrows), dpi=100)
    im1 = axs.imshow(sim_W, cmap='viridis', interpolation='nearest', vmin=sim_W_min, vmax=sim_W_max)
    # axs.set_title(f'W Pairwise similarity', fontsize=12)
    axs.set_yticks(range(len(labels)))
    axs.set_yticklabels(labels, fontsize=ftsz)
    axs.set_xticks(range(len(labels)))
    axs.set_xticklabels(labels, rotation=90, fontsize=ftsz)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    cbar = plt.colorbar(im1,)
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(f'{figure_dir}/W_sim_label_{model_name}_{dataset_name}.pdf')
    print(f'{figure_dir}/W_sim_label_{model_name}_{dataset_name}.pdf')
    # plt.show()
    # save/load labels
    with open("w_labels.txt", "w") as file:
        for item in labels:
            file.write("%s\n" % item)


W_sim_plot_with_label()
# exit(0)

def H_sim_plot_with_label():
    H = H_list[-1]
    uniq_emb = tsk.get_unique()
    labels = []
    for i in range(len(uniq_emb)):
        lbl = tsk.v_ctx2v[uniq_emb[i]]
        lbl_str = [sp.IdToPiece(int(id_)) for id_ in list(lbl)]
        lbl_str = "".join(lbl_str)
        print(lbl)
        print(i, lbl_str)
        labels.append(lbl_str)

    sim_H = pairwise_cossim(H)
    sim_H_min, sim_H_max = min(sim_H.flatten()), max(sim_H.flatten())
    nrows, ncols = 1, 1
    ftsz = 8
    fig, axs = plt.subplots(nrows, ncols, figsize=(60 * ncols, 50 * nrows), dpi=100)
    im1 = axs.imshow(sim_H, cmap='viridis', interpolation='nearest', vmin=sim_H_min, vmax=sim_H_max)
    axs.set_title(f'H Pairwise_similarity', fontsize=30)
    axs.set_yticks(range(len(labels)))
    axs.set_yticklabels(labels, fontsize=ftsz)
    axs.set_xticks(range(len(labels)))
    axs.set_xticklabels(labels, rotation=90, fontsize=ftsz)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    cbar = plt.colorbar(im1,)
    cbar.ax.tick_params(labelsize=30)
    plt.savefig(f'{figure_dir}/H_sim_label_{model_name}_{dataset_name}.pdf')
    # plt.show()
    with open("h_labels.txt", "w") as file:
        for item in labels:
            file.write("%s\n" % item)

H_sim_plot_with_label()

exit(0)

def H_sim_plot_selected_with_label():
    H = H_list[-1]
    # select one per 3 for the plot
    sel = np.arange(0, H.shape[0], 3)
    H_sel = H[sel]
    uniq_emb = tsk.get_unique()
    labels = []
    for i in range(len(sel)):
        lbl = tsk.v_ctx2v[uniq_emb[sel[i]]]
        lbl_str = [sp.IdToPiece(int(id_)) for id_ in list(lbl)]
        lbl_str = "".join(lbl_str)
        print(lbl)
        print(i, lbl_str)
        labels.append(lbl_str)

    sim_H = pairwise_cossim(H_sel)
    sim_H_min, sim_H_max = min(sim_H.flatten()), max(sim_H.flatten())
    nrows, ncols = 1, 1
    ftsz = 18
    fig, axs = plt.subplots(nrows, ncols, figsize=(60 * ncols, 50 * nrows), dpi=100)
    im1 = axs.imshow(sim_H, cmap='viridis', interpolation='nearest', vmin=sim_H_min, vmax=sim_H_max)
    axs.set_title(f'H Pairwise_similarity', fontsize=30)
    axs.set_yticks(range(len(labels)))
    axs.set_yticklabels(labels, fontsize=ftsz)
    axs.set_xticks(range(len(labels)))
    axs.set_xticklabels(labels, rotation=90, fontsize=ftsz)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    cbar = plt.colorbar(im1,)
    cbar.ax.tick_params(labelsize=30)
    plt.savefig(f'{figure_dir}/H_sim_selected_label_{model_name}_{dataset_name}.pdf')
    # plt.show()

# H_sim_plot_selected_with_label()
# exit(0)

def compute_structural_component(x, y):
    if x.shape != y.shape:
        raise ValueError("dimensions of x and y do not match, x.shape = {}, y.shape = {}".format(x.shape, y.shape))

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = np.std(x, ddof=1)
    sigma_y = np.std(y, ddof=1)

    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    C = 1e-5

    structural_component = (sigma_xy + C) / (sigma_x * sigma_y + C)
    return structural_component

corr_dir = "/Users/Lenovo/Next_Token/rebuttal/corr"

GHmm_GS_corr = compute_structural_component(HG_mm_thm, S@S.T)
GWmm_GS_corr = compute_structural_component(WG_mm_thm, S.T@S)

print(f'GHmm_GS_corr: {GHmm_GS_corr}')
print(f'GWmm_GS_corr: {GWmm_GS_corr}')
def plot_GH_sqrtLmm_structural_corr():
    structure_corr = []
    for h in H_list:
        # hsim = h @ h.T
        hsim = pairwise_cossim(h)
        structure_corr.append(compute_structural_component(hsim, sqrt_GL_mm_thm))

    np.save(f'{corr_dir}/structure_corr_gh_sqrtlmm_{dataset_name}_{model_name}_d{dim}.npy', structure_corr)
    # plot the curve
    plt.figure(figsize=(8,4))
    plt.plot(rec, np.array(structure_corr))
    plt.xlabel(r'Epoch $k$')
    plt.ylabel('Structural Correlation')
    # plot in log scale
    plt.xscale('log')
    plt.title(f'Structural Correlation between Hgram and sqrt(Lmm), {dataset_name}, {model_name}, d{dim} ')
    plt.savefig(f'{figure_dir}/GH_sqrtLmm_structural_corr_{dataset_name}_{model_name}_d{dim}.pdf')

def plot_GW_sqrtLmm_structural_corr():
    structure_corr = []
    for w in W_list:
        wsim = w @ w.T
        structure_corr.append(compute_structural_component(wsim, sqrt_GL_mm_thm_T))

    np.save(f'{corr_dir}/structure_corr_gw_sqrtlmm_{dataset_name}_{model_name}_d{dim}.npy', structure_corr)
    # plot the curve
    plt.figure(figsize=(8,4))
    plt.plot(rec, np.array(structure_corr))
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    # plot in log scale
    plt.xscale('log')
    plt.title(f'Structural Correlation between Wgram and sqrt(Lmm.T), {dataset_name}, {model_name}, d{dim} ')
    plt.savefig(f'{figure_dir}/GW_sqrtLmm_structural_corr_{dataset_name}_{model_name}_d{dim}.pdf')

# plot_GW_sqrtLmm_structural_corr()
# plot_GH_sqrtLmm_structural_corr()

def plot_W_S_sim_structual_corr():
    structure_corr = []
    ssim = pairwise_cossim(S.T)
    for w in W_list:
        wsim = pairwise_cossim(w)
        # ssim = pairwise_cossim(S)
        structure_corr.append(compute_structural_component(wsim, ssim))

    np.save(f'{corr_dir}/structure_corr_w_{dataset_name}_{model_name}_d{dim}.npy', structure_corr)
    np.save(f'{corr_dir}/rectemp_w_{dataset_name}_{model_name}_d{dim}.npy', rec)
    # plot the curve
    plt.figure(figsize=(8,4))
    plt.plot(rec, np.array(structure_corr))
    plt.xlabel(r'Epoch $k$')
    plt.ylabel('Structural Correlation')
    # plot in log scale
    plt.xscale('log')
    plt.title(f'Structural Correlation between Cos(W, W) and Cos(S, S), {dataset_name}, {model_name}, d{dim} ')
    plt.savefig(f'{figure_dir}/W_S_sim_structural_corr_{dataset_name}_{model_name}_d{dim}.pdf')
    # plt.show()

# plot_W_S_sim_structual_corr()

def plot_HW_S_sim_structual_corr():
    structure_corr_ufm404_h = []
    structure_corr_tf404_h = []
    structure_corr_ufmsmall_h = []
    structure_corr_tfsmall_h = []
    structure_corr_ufm404_w = []
    structure_corr_tf404_w = []
    structure_corr_ufmsmall_w = []
    structure_corr_tfsmall_w = []

    for i in range(len(H_list_ufmsmall)):
        h = H_list_ufmsmall[i]
        w = W_list_ufmsmall[i]
        hsim = pairwise_cossim(h)
        ssim = pairwise_cossim(S_tfsmall)
        wsim = pairwise_cossim(w)
        ssimt = pairwise_cossim(S_tfsmall.T)

        structure_corr_ufm404_h.append(compute_structural_component(hsim, ssim))
        structure_corr_ufm404_w.append(compute_structural_component(wsim, ssimt))

    for i in range(len(H_list_tfsmall)):
        h = H_list_tfsmall[i]
        w = W_list_tfsmall[i]
        hsim = pairwise_cossim(h)
        ssim = pairwise_cossim(S_tfsmall)
        wsim = pairwise_cossim(w)
        ssimt = pairwise_cossim(S_tfsmall.T)

        structure_corr_tfsmall_h.append(compute_structural_component(hsim, ssim))
        structure_corr_tfsmall_w.append(compute_structural_component(wsim, ssimt))

    for i in range(len(H_list_ufm404)):
        h = H_list_ufm404[i]
        w = W_list_ufm404[i]
        hsim = pairwise_cossim(h)
        ssim = pairwise_cossim(S_tf404)
        wsim = pairwise_cossim(w)
        ssimt = pairwise_cossim(S_tf404.T)

        structure_corr_ufmsmall_h.append(compute_structural_component(hsim, ssim))
        structure_corr_ufmsmall_w.append(compute_structural_component(wsim, ssimt))

    for i in range(len(H_list_tf404)):
        h = H_list_tf404[i]
        w = W_list_tf404[i]
        hsim = pairwise_cossim(h)
        ssim = pairwise_cossim(S_tf404)
        wsim = pairwise_cossim(w)
        ssimt = pairwise_cossim(S_tf404.T)

        structure_corr_tf404_h.append(compute_structural_component(hsim, ssim))
        structure_corr_tf404_w.append(compute_structural_component(wsim, ssimt))

    # plot the curves
    mksz = 4
    plt.figure(figsize=(6, 4))
    plt.plot(rec_tf404, structure_corr_tf404_w,'b-o', markersize=mksz, label='Simplified TinyStories, TF, ssim w')
    plt.plot(rec_tf404, structure_corr_tf404_h, 'b-v',markersize=mksz, label='Simplified TinyStories, TF, ssim h')
    plt.plot(rec_ufm404, structure_corr_ufm404_w, 'b--o',markersize=mksz, label='Simplified TinyStories, UFM, ssim w')
    plt.plot(rec_ufm404, structure_corr_ufm404_h, 'b--v',markersize=mksz, label='Simplified TinyStories, UFM, ssim h')
    plt.plot(rec_tfsmall, structure_corr_tfsmall_w, 'r-o',markersize=mksz, label='Very Small, TF, ssim w')
    plt.plot(rec_tfsmall, structure_corr_tfsmall_h, 'r-v',markersize=mksz, label='Very Small, TF, ssim h')
    plt.plot(rec_ufmsmall, structure_corr_ufmsmall_w, 'r--o',markersize=mksz, label='Very Small, UFM, ssim w')
    plt.plot(rec_ufmsmall, structure_corr_ufmsmall_h, 'r--v',markersize=mksz, label='Very Small, UFM, ssim h')
    plt.xlabel(r'Epoch $k$')
    # plt.ylabel('Structural Correlation')
    # plt.ylim(0.5, 1)
    # plot in log scale
    plt.xscale('log')
    plt.grid()
    plt.xlim(1, 30000)
    # plt.yscale('log')

    custom_lines = [
        Line2D([0], [0,1], color='r',),
        Line2D([0], [0,1], color='b',),
        Line2D([0], [0,1], color='k', linestyle='-'),
        Line2D([0], [0,1], color='k', linestyle='--'),
        Line2D([0], [0, 1], color='k', marker='o'),
        Line2D([0], [0, 1], color='k', marker='v')
    ]

    custom_labels = [
        'Synthetic',
        'Simplified TinyStories',
        'TF',
        'UFM',
        'W',
        'H'
    ]

    plt.legend(custom_lines, custom_labels)
    # plt.title(f'Structural Correlation between H.TH and S.TS, on two small scale datasets')
    plt.savefig(f'{figure_dir}/H_S_sim_structural_corr_{dataset_name}_{model_name}_d{dim}.pdf')

# plot_HW_S_sim_structual_corr()

def plot_HW_Lmmthm_sim_structual_corr():
    structure_corr_ufm404_h = []
    structure_corr_tf404_h = []
    structure_corr_ufmsmall_h = []
    structure_corr_tfsmall_h = []
    structure_corr_ufm404_w = []
    structure_corr_tf404_w = []
    structure_corr_ufmsmall_w = []
    structure_corr_tfsmall_w = []

    Lmm_thm_small = np.load(f'{tfm_verysmall_result}/Lmm_cvxpy_verysmallset.npy')
    Lmm_thm_404 = np.load(f'{ufm_tiny404_result}/Lmm_cvxpy_tiny_extract_m404.npy')

    for i in range(len(H_list_ufmsmall)):
        h = H_list_ufmsmall[i]
        w = W_list_ufmsmall[i]
        hsim = pairwise_cossim(h)
        lmm_sim = pairwise_cossim(Lmm_thm_small)
        lmmt_sim = pairwise_cossim(Lmm_thm_small.T)
        wsim = pairwise_cossim(w)


        structure_corr_ufm404_h.append(compute_structural_component(hsim, lmm_sim))
        structure_corr_ufm404_w.append(compute_structural_component(wsim, lmmt_sim))

    for i in range(len(H_list_tfsmall)):
        h = H_list_tfsmall[i]
        w = W_list_tfsmall[i]
        hsim = pairwise_cossim(h)
        lmm_sim = pairwise_cossim(Lmm_thm_small)
        lmmt_sim = pairwise_cossim(Lmm_thm_small.T)
        wsim = pairwise_cossim(w)

        structure_corr_tfsmall_h.append(compute_structural_component(hsim, lmm_sim))
        structure_corr_tfsmall_w.append(compute_structural_component(wsim, lmmt_sim))

    for i in range(len(H_list_ufm404)):
        h = H_list_ufm404[i]
        w = W_list_ufm404[i]
        hsim = pairwise_cossim(h)
        lmm_sim = pairwise_cossim(Lmm_thm_404)
        lmmt_sim = pairwise_cossim(Lmm_thm_404.T)
        wsim = pairwise_cossim(w)

        structure_corr_ufmsmall_h.append(compute_structural_component(hsim, lmm_sim))
        structure_corr_ufmsmall_w.append(compute_structural_component(wsim, lmmt_sim))

    for i in range(len(H_list_tf404)):
        h = H_list_tf404[i]
        w = W_list_tf404[i]
        hsim = pairwise_cossim(h)
        lmm_sim = pairwise_cossim(Lmm_thm_404)
        lmmt_sim = pairwise_cossim(Lmm_thm_404.T)
        wsim = pairwise_cossim(w)

        structure_corr_tf404_h.append(compute_structural_component(hsim, lmm_sim))
        structure_corr_tf404_w.append(compute_structural_component(wsim, lmmt_sim))

    # plot the curves
    # mksz = 4
    # plt.figure(figsize=(6, 4))
    # plt.plot(rec_tf404, structure_corr_tf404_w,'b-o', markersize=mksz, label='Simplified TinyStories, TF, ssim w')
    # plt.plot(rec_tf404, structure_corr_tf404_h, 'b-v',markersize=mksz, label='Simplified TinyStories, TF, ssim h')
    # plt.plot(rec_ufm404, structure_corr_ufm404_w, 'b--o',markersize=mksz, label='Simplified TinyStories, UFM, ssim w')
    # plt.plot(rec_ufm404, structure_corr_ufm404_h, 'b--v',markersize=mksz, label='Simplified TinyStories, UFM, ssim h')
    # plt.plot(rec_tfsmall, structure_corr_tfsmall_w, 'r-o',markersize=mksz, label='Very Small, TF, ssim w')
    # plt.plot(rec_tfsmall, structure_corr_tfsmall_h, 'r-v',markersize=mksz, label='Very Small, TF, ssim h')
    # plt.plot(rec_ufmsmall, structure_corr_ufmsmall_w, 'r--o',markersize=mksz, label='Very Small, UFM, ssim w')
    # plt.plot(rec_ufmsmall, structure_corr_ufmsmall_h, 'r--v',markersize=mksz, label='Very Small, UFM, ssim h')
    # plt.xlabel('Epoch')
    # # plt.ylabel('Structural Correlation')
    # # plt.ylim(0.5, 1)
    # # plot in log scale
    # plt.xscale('log')
    # plt.xlim(1, 30000)
    # # plt.yscale('log')
    #
    # custom_lines = [
    #     Line2D([0], [0,1], color='r',),
    #     Line2D([0], [0,1], color='b',),
    #     Line2D([0], [0,1], color='k', linestyle='-'),
    #     Line2D([0], [0,1], color='k', linestyle='--'),
    #     Line2D([0], [0, 1], color='k', marker='o'),
    #     Line2D([0], [0, 1], color='k', marker='v')
    # ]
    #
    # custom_labels = [
    #     'Synthetic',
    #     'Simplified TinyStories',
    #     'TF',
    #     'UFM',
    #     'W',
    #     'H'
    # ]
    # plot the curves
    # mksz = 4
    plt.figure(figsize=(6, 4))
    plt.plot(rec_tf404, structure_corr_tf404_w,color='steelblue',linestyle='--', label='Simplified TinyStories, TF, ssim w')
    plt.plot(rec_tf404, structure_corr_tf404_h, color='steelblue',label='Simplified TinyStories, TF, ssim h')
    plt.plot(rec_ufm404, structure_corr_ufm404_w, color='deepskyblue',linestyle='--', label='Simplified TinyStories, UFM, ssim w')
    plt.plot(rec_ufm404, structure_corr_ufm404_h, color='deepskyblue',label='Simplified TinyStories, UFM, ssim h')
    plt.plot(rec_tfsmall, structure_corr_tfsmall_w,color='darkorange', linestyle='--',label='Very Small, TF, ssim w')
    plt.plot(rec_tfsmall, structure_corr_tfsmall_h, color='darkorange', label='Very Small, TF, ssim h')
    plt.plot(rec_ufmsmall, structure_corr_ufmsmall_w, color='gold',linestyle='--', label='Very Small, UFM, ssim w')
    plt.plot(rec_ufmsmall, structure_corr_ufmsmall_h, color='gold',label='Very Small, UFM, ssim h')
    plt.xlabel('Epoch')

    custom_lines = [
        Line2D([0], [0,1], color='gold',),
        Line2D([0], [0,1], color='darkorange',),
        Line2D([0], [0,1], color='deepskyblue'),
        Line2D([0], [0,1], color='steelblue'),
        Line2D([0], [0, 1], color='k', linestyle='-'),
        Line2D([0], [0, 1], color='k', linestyle='--'),
    ]

    custom_labels = [
        'Synthetic, UFM',
        'Synthetic, TF',
        'Simplified TinyStories, UFM',
        'Simplified TinyStories, TF',
        # r'$SIM(CORR(H),CORR(\overline{S}))$',
        # r'$SIM(CORR(W^T),CORR(\overline{S}^T))$'
        'H',
        'W'
    ]
    plt.xlabel(r'Epoch $k$')
    # plt.ylabel('Structural Correlation')
    # plt.ylim(0.5, 1)
    # plot in log scale
    plt.xscale('log')
    plt.grid()
    plt.xlim(1, 30000)
    # plt.yscale('log')

    plt.legend(custom_lines, custom_labels)
    # plt.title(f'Structural Correlation between H.TH and S.TS, on two small scale datasets')
    plt.savefig(f'{figure_dir}/HW_Lmmthm_sim_structural_corr_{dataset_name}_{model_name}_d{dim}.pdf')

# curve
# plot_HW_Lmmthm_sim_structual_corr()

def plot_HW_HmmWmm_sim_structual_corr():
    structure_corr_ufm404_h = []
    structure_corr_tf404_h = []
    structure_corr_ufmsmall_h = []
    structure_corr_tfsmall_h = []
    structure_corr_ufm404_w = []
    structure_corr_tf404_w = []
    structure_corr_ufmsmall_w = []
    structure_corr_tfsmall_w = []

    L_mm_cvxpy_small = np.load(f'{ufm_verysmall_result}/Lmm_cvxpy_verysmallset.npy')
    L_mm_cvxpy_404 = np.load(f'{ufm_tiny404_result}/Lmm_cvxpy_tiny_extract_m404.npy')
    WG_mm_norm_cvxpy_small, HG_mm_norm_cvxpy_small, W_mm_cvxpy_small, H_mm_cvxpy_small, u_cvxpy, vh_cvxpy, WG_mm_cvxpy, HG_mm_cvxpy = compute_grams(
        L_mm_cvxpy_small)
    WG_mm_norm_cvxpy_404, HG_mm_norm_cvxpy_404, W_mm_cvxpy_404, H_mm_cvxpy_404, u_cvxpy, vh_cvxpy, WG_mm_cvxpy, HG_mm_cvxpy = compute_grams(
        L_mm_cvxpy_404)

    # compute Hmm ans Wmm

    for i in range(len(H_list_ufmsmall)):
        h = H_list_ufmsmall[i]
        w = W_list_ufmsmall[i]
        hsim = pairwise_cossim(h)
        hmmsim = pairwise_cossim(H_mm_cvxpy_small)
        wsim = pairwise_cossim(w)
        wmmsim = pairwise_cossim(W_mm_cvxpy_small)

        # print(compute_structural_component(hsim, hmmsim))
        structure_corr_ufmsmall_h.append(compute_structural_component(hsim, hmmsim))
        structure_corr_ufmsmall_w.append(compute_structural_component(wsim, wmmsim))

    for i in range(len(H_list_tfsmall)):
        h = H_list_tfsmall[i]
        w = W_list_tfsmall[i]
        hsim = pairwise_cossim(h)
        hmmsim = pairwise_cossim(H_mm_cvxpy_small)
        wsim = pairwise_cossim(w)
        wmmsim = pairwise_cossim(W_mm_cvxpy_small)

        structure_corr_tfsmall_h.append(compute_structural_component(hsim, hmmsim))
        structure_corr_tfsmall_w.append(compute_structural_component(wsim, wmmsim))

    for i in range(len(H_list_ufm404)):
        h = H_list_ufm404[i]
        w = W_list_ufm404[i]
        hsim = pairwise_cossim(h)
        hmmsim = pairwise_cossim(H_mm_cvxpy_404)
        wsim = pairwise_cossim(w)
        wmmsim = pairwise_cossim(W_mm_cvxpy_404)

        structure_corr_ufm404_h.append(compute_structural_component(hsim, hmmsim))
        structure_corr_ufm404_w.append(compute_structural_component(wsim, wmmsim))

    for i in range(len(H_list_tf404)):
        h = H_list_tf404[i]
        w = W_list_tf404[i]
        hsim = pairwise_cossim(h)
        hmmsim = pairwise_cossim(H_mm_cvxpy_404)
        wsim = pairwise_cossim(w)
        wmmsim = pairwise_cossim(W_mm_cvxpy_404)

        structure_corr_tf404_h.append(compute_structural_component(hsim, hmmsim))
        structure_corr_tf404_w.append(compute_structural_component(wsim, wmmsim))

    # plot the curves
    # mksz = 4
    # plt.figure(figsize=(6, 4))
    # plt.plot(rec_tf404, structure_corr_tf404_w,'b-o', markersize=mksz, label='Simplified TinyStories, TF, ssim w')
    # plt.plot(rec_tf404, structure_corr_tf404_h, 'b-v',markersize=mksz, label='Simplified TinyStories, TF, ssim h')
    # plt.plot(rec_ufm404, structure_corr_ufm404_w, 'b--o',markersize=mksz, label='Simplified TinyStories, UFM, ssim w')
    # plt.plot(rec_ufm404, structure_corr_ufm404_h, 'b--v',markersize=mksz, label='Simplified TinyStories, UFM, ssim h')
    # plt.plot(rec_tfsmall, structure_corr_tfsmall_w, 'r-o',markersize=mksz, label='Very Small, TF, ssim w')
    # plt.plot(rec_tfsmall, structure_corr_tfsmall_h, 'r-v',markersize=mksz, label='Very Small, TF, ssim h')
    # plt.plot(rec_ufmsmall, structure_corr_ufmsmall_w, 'r--o',markersize=mksz, label='Very Small, UFM, ssim w')
    # plt.plot(rec_ufmsmall, structure_corr_ufmsmall_h, 'r--v',markersize=mksz, label='Very Small, UFM, ssim h')
    # plt.xlabel('Epoch')
    # # plt.ylabel('Structural Correlation')
    # # plt.ylim(0.5, 1)
    # # plot in log scale
    # plt.xscale('log')
    # plt.xlim(1, 30000)
    # # plt.yscale('log')
    #
    # custom_lines = [
    #     Line2D([0], [0,1], color='r',),
    #     Line2D([0], [0,1], color='b',),
    #     Line2D([0], [0,1], color='k', linestyle='-'),
    #     Line2D([0], [0,1], color='k', linestyle='--'),
    #     Line2D([0], [0, 1], color='k', marker='o'),
    #     Line2D([0], [0, 1], color='k', marker='v')
    # ]
    #
    # custom_labels = [
    #     'Synthetic',
    #     'Simplified TinyStories',
    #     'TF',
    #     'UFM',
    #     'W',
    #     'H'
    # ]


    # plot the curves
    mksz = 4
    plt.figure(figsize=(6, 4))
    plt.plot(rec_tf404+1, structure_corr_tf404_w,color='steelblue', linestyle='--', markersize=mksz, label='Simplified TinyStories, TF, ssim w')
    plt.plot(rec_tf404+1, structure_corr_tf404_h,color='steelblue',markersize=mksz, label='Simplified TinyStories, TF, ssim h')
    plt.plot(rec_ufm404+1, structure_corr_ufm404_w,color='deepskyblue', linestyle='--',markersize=mksz, label='Simplified TinyStories, UFM, ssim w')
    plt.plot(rec_ufm404+1, structure_corr_ufm404_h,color='deepskyblue',markersize=mksz, label='Simplified TinyStories, UFM, ssim h')
    plt.plot(rec_tfsmall+1, structure_corr_tfsmall_w,color='darkorange', linestyle='--',markersize=mksz, label='Very Small, TF, ssim w')
    plt.plot(rec_tfsmall+1, structure_corr_tfsmall_h,color='darkorange',markersize=mksz, label='Very Small, TF, ssim h')
    plt.plot(rec_ufmsmall+1, structure_corr_ufmsmall_w, color='gold', linestyle='--',markersize=mksz, label='Very Small, UFM, ssim w')
    plt.plot(rec_ufmsmall+1, structure_corr_ufmsmall_h,color='gold',markersize=mksz, label='Very Small, UFM, ssim h')
    plt.xlabel(r'Epoch $k$')
    # plt.ylabel('Structural Correlation')
    # plt.ylim(0.5, 1)
    # plot in log scale
    plt.xscale('log')
    plt.xlim(1, 30000)
    # plt.yscale('log')
    plt.grid()


    custom_lines = [
        Line2D([0], [0,1], color='gold',),
        Line2D([0], [0,1], color='darkorange',),
        Line2D([0], [0,1], color='deepskyblue'),
        Line2D([0], [0,1], color='steelblue'),
        Line2D([0], [0, 1], color='k', linestyle='-'),
        Line2D([0], [0, 1], color='k', linestyle='--'),
    ]

    custom_labels = [
        'Synthetic, UFM',
        'Synthetic, TF',
        'Simplified TinyStories, UFM',
        'Simplified TinyStories, TF',
        # r'$SIM(CORR(H),CORR((H_{mm}))$',
        # r'$SIM(CORR(W^T),CORR(W_{mm}^T))$'
        'H',
        'W'
    ]

    plt.legend(custom_lines, custom_labels)
    # plt.title(f'Structural Correlation between H.TH and S.TS, on two small scale datasets')
    plt.savefig(f'{figure_dir}/HW_HWmm_sim_structural_corr_{dataset_name}_{model_name}_d{dim}.pdf')
# curve
# plot_HW_Hmm Wmm_sim_structual_corr()
exit(0)


def save_H_S_gram_structual_corr():
    structure_corr_hg = []
    structure_corr_wg = []
    for i in range(len(H_list)):
        h = H_list[i]
        w = W_list[i]
        # hsim = pairwise_cossim(h)
        # ssim = pairwise_cossim(S)
        hsim = h @ h.T
        ssim = S @ S.T
        wsim = w @ w.T
        ssimt = S.T @ S
        structure_corr_hg.append(compute_structural_component(hsim, ssim))
        structure_corr_wg.append(compute_structural_component(wsim, ssimt))

    np.save(f'{corr_dir}/structure_corr_hg_{dataset_name}_{model_name}_d{dim}.npy', structure_corr_hg)
    np.save(f'{corr_dir}/structure_corr_wg_{dataset_name}_{model_name}_d{dim}.npy', structure_corr_wg)

def save_gwh_glmm_structural_corr():
    structure_corr_gwlmm = []
    structure_corr_ghlmm = []
    for i in range(len(W_list)):
        w = W_list[i]
        h = H_list[i]
        # wsim = w @ w.T
        wsim = pairwise_cossim(w)
        lmm_sim = pairwise_cossim(L_mm_thm)
        # hsim = h @ h.T
        hsim = pairwise_cossim(h)
        lmmt_sim = pairwise_cossim(L_mm_thm.T)
        # structure_corr_gwlmm.append(compute_structural_component(wsim, GL_mm_thm_T))
        # structure_corr_ghlmm.append(compute_structural_component(hsim, GL_mm_thm))
        structure_corr_gwlmm.append(compute_structural_component(wsim, lmmt_sim))
        structure_corr_ghlmm.append(compute_structural_component(hsim, lmm_sim))

    np.save(f'{corr_dir}/structure_corr_gwlmm_sim_{dataset_name}_{model_name}_d{dim}.npy', structure_corr_gwlmm)
    np.save(f'{corr_dir}/structure_corr_ghlmm_sim_{dataset_name}_{model_name}_d{dim}.npy', structure_corr_ghlmm)

save_gwh_glmm_structural_corr()

# save_H_S_gram_structual_corr()
def plot_hg_structual_corr_bothdatasets():
    # load the structure_corr
    structure_corr_404 = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_hg_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    rec = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/rectemp_w_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    structure_corr_small = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_hg_verysmallset_4 layer Transformer pos off_d128.npy')
    # plot the curves
    plt.figure(figsize=(8, 4))
    plt.plot(rec, np.array(structure_corr_404), label='Synthetic data from TinyStories')
    plt.plot(rec, np.array(structure_corr_small), label='Small synthetic dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    plt.ylim(0.5, 1)
    # plot in log scale
    plt.xscale('log')
    plt.legend()

    plt.title(f'Structural Correlation between H.TH and S.TS, on two small scale datasets')
    plt.savefig(f'{figure_dir}/hg_structural_corr_bothdatasets.pdf')
    # plt.show()

def plot_wg_structual_corr_bothdatasets():
    # load the structure_corr
    structure_corr_404 = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_wg_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    rec = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/rectemp_w_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    structure_corr_small = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_wg_verysmallset_4 layer Transformer pos off_d128.npy')
    # plot the curves
    plt.figure(figsize=(8, 4))
    plt.plot(rec, np.array(structure_corr_404), label='Synthetic data from TinyStories')
    plt.plot(rec, np.array(structure_corr_small), label='Small synthetic dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    plt.ylim(0., 1)
    # plot in log scale
    plt.xscale('log')
    plt.legend()
    plt.title(f'Structural Correlation between WW.T and SS.T, on two small scale datasets')
    plt.savefig(f'{figure_dir}/wg_structural_corr_bothdatasets.pdf')
    # plt.show()

def plot_hglmm_structual_corr_bothdatasets():
    # load the structure_corr
    structure_corr_404 = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_ghlmm_sim_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    rec = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/rectemp_w_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    structure_corr_small = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_ghlmm_sim_verysmallset_4 layer Transformer pos off_d128.npy')
    # plot the curves
    plt.figure(figsize=(8, 4))
    plt.plot(rec, np.array(structure_corr_404), label='Synthetic data from TinyStories')
    plt.plot(rec, np.array(structure_corr_small), label='Small synthetic dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    plt.ylim(0.5, 1)
    # plot in log scale
    plt.xscale('log')
    plt.legend()
    plt.title(f'Structural Correlation between $cos(H,H)$ and $cos((I-1/V11^T)S,(I-1/V11^T)S)$ thm,, on two small scale datasets')
    plt.savefig(f'{figure_dir}/ghlmm_sim_structural_corr_bothdatasets.pdf')


def plot_wglmm_structual_corr_bothdatasets():
    # load the structure_corr
    structure_corr_404 = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_gwlmm_sim_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    rec = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/rectemp_w_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    structure_corr_small = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_gwlmm_sim_verysmallset_4 layer Transformer pos off_d128.npy')
    # plot the curves
    plt.figure(figsize=(8, 4))
    plt.plot(rec, np.array(structure_corr_404), label='Synthetic data from TinyStories')
    plt.plot(rec, np.array(structure_corr_small), label='Small synthetic dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    plt.ylim(0., 1)
    # plot in log scale
    plt.xscale('log')
    plt.legend()
    plt.title(f'Structural Correlation between $cos(W^T,W^T)$ and $cos(S^T(I-1/V11^T),S^T(I-1/V11^T))$, on two small scale datasets')
    plt.savefig(f'{figure_dir}/gwlmm_sim_structural_corr_bothdatasets.pdf')
    # plt.show()


# do this for tf on both datasets on one plot
def plot_H_S_sim_structual_corr_bothdatasets():

    # load the structure_corr
    structure_corr_404 = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    rec = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/rectemp_verysmallset_4 layer Transformer pos off_d128.npy')
    structure_corr_small = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_verysmallset_4 layer Transformer pos off_d128.npy')
    # plot the curves
    plt.figure(figsize=(8, 4))
    plt.plot(rec, np.array(structure_corr_404), label='Synthetic data from TinyStories')
    plt.plot(rec, np.array(structure_corr_small), label='Small synthetic dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    plt.ylim(0.5, 1)

    # plot in log scale
    plt.xscale('log')
    plt.legend()
    plt.title(f'Structural Correlation between Cos(H, H) and Cos(S, S), on two small scale datasets')
    plt.savefig(f'{figure_dir}/H_S_sim_structural_corr_bothdatasets.pdf')
    # plt.show()

# plot_H_S_gram_structual_corr_bothdatasets()

def plot_W_S_sim_structual_corr_bothdatasets():

    # load the structure_corr
    structure_corr_404 = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_w_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    rec = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/rectemp_w_verysmallset_4 layer Transformer pos off_d128.npy')
    structure_corr_small = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_w_verysmallset_4 layer Transformer pos off_d128.npy')
    # plot the curves
    plt.figure(figsize=(8, 4))
    plt.plot(rec, np.array(structure_corr_404), label='Synthetic data from TinyStories')
    plt.plot(rec, np.array(structure_corr_small), label='Small synthetic dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    plt.ylim(0., 1)

    # plot in log scale
    plt.xscale('log')
    plt.legend()
    plt.title(f'Structural Correlation between $Cos(W^T, W^T)$ and $Cos(S^T, S^T)$, on two small scale datasets')
    plt.savefig(f'{figure_dir}/W_S_sim_structural_corr_bothdatasets.pdf')
    # plt.show()

# plot_W_S_gram_structual_corr_bothdatasets()

def plot_gw_sqrtlmm_structural_corr_bothdatasets():

    # load the structure_corr
    structure_corr_404 = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_gw_sqrtlmm_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
    rec = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/rectemp_w_verysmallset_4 layer Transformer pos off_d128.npy')
    structure_corr_small = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_gw_sqrtlmm_verysmallset_4 layer Transformer pos off_d128.npy')
    # plot the curves
    plt.figure(figsize=(8, 4))
    plt.plot(rec, np.array(structure_corr_404), label='Synthetic data from TinyStories')
    plt.plot(rec, np.array(structure_corr_small), label='Small synthetic dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Structural Correlation')
    plt.ylim(0., 1)

    # plot in log scale
    plt.xscale('log')
    plt.legend()
    plt.title(f'Structural Correlation between $WW^T$ and $gLmm.T$ thm, on two small scale datasets')
    plt.savefig(f'{figure_dir}/gw_sqrtlmm_structural_corr_bothdatasets.pdf')
    # plt.show()

def plot_gh_sqrtlmm_structural_corr_bothdatasets():

        # load the structure_corr
        structure_corr_404 = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_gh_sqrtlmm_tiny_extract_m404_4 layer Transformer pos off_d128.npy')
        rec = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/rectemp_w_verysmallset_4 layer Transformer pos off_d128.npy')
        structure_corr_small = np.load(f'/Users/Lenovo/Next_Token/rebuttal/corr/structure_corr_gh_sqrtlmm_verysmallset_4 layer Transformer pos off_d128.npy')
        # plot the curves
        plt.figure(figsize=(8, 4))
        plt.plot(rec, np.array(structure_corr_404), label='Synthetic data from TinyStories')
        plt.plot(rec, np.array(structure_corr_small), label='Small synthetic dataset')
        plt.xlabel('Epoch')
        plt.ylabel('Structural Correlation')
        plt.ylim(0.5, 1)

        # plot in log scale
        plt.xscale('log')
        plt.legend()
        plt.title(f'Structural Correlation between $H.TH$ and $gLmm$ thm, on two small scale datasets')
        plt.savefig(f'{figure_dir}/gh_sqrtlmm_structural_corr_bothdatasets.pdf')
        # plt.show()

# plot_gh_sqrtlmm_structural_corr_bothdatasets()
# plot_gw_sqrtlmm_structural_corr_bothdatasets()
# plot_H_S_sim_structual_corr_bothdatasets()
# plot_W_S_sim_structual_corr_bothdatasets()
# plot_hg_structual_corr_bothdatasets()
# plot_wg_structual_corr_bothdatasets()
# plot_wglmm_structual_corr_bothdatasets()
# plot_hglmm_structual_corr_bothdatasets()
# H_sim_plot_with_label()
# # generate plot with Lmm cvxpy
# lmm_name="cvxpy"
# matrix_plot(WG_mm_norm_cvxpy, HG_mm_norm_cvxpy,L_mm_cvxpy_normed,lmm_name)
# embedding_similarity_plot(H_mm_cvxpy, lmm_name)
# matrix_plot_all(H_mm_cvxpy, W_mm_cvxpy,L_mm_cvxpy, lmm_name)
# Sim_plot_all(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, lmm_name)
# Sim_plot_iter(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, "cvxpy")
# redo_figure3(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, "cvxpy")
# # generate plot with Lmm thm
# lmm_name="theorem"
# matrix_plot(WG_mm_norm_thm, HG_mm_norm_thm,L_mm_thm_normed,lmm_name)
# embedding_similarity_plot(H_mm_thm, lmm_name)
# matrix_plot_all(H_mm_thm, W_mm_thm,L_mm_thm, lmm_name)
# Sim_plot_all(H_mm_thm, W_mm_thm, L_mm_thm, lmm_name)








