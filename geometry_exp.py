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
from itertools import combinations
from scipy.optimize import linprog



from Dataset import *
from utils import *
from balls import *
from cube import *
from cones import *



import torch
from Model import Transformer, FixedLengthModelArgs, FixedLengthMLP, MultiLabelSVM, TFMModelArgs, UFM
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics.pairwise import cosine_similarity



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


# traininf_tag = "_cpu_verysmall_tfm_128_fix_s3"
# traininf_tag = "_cpu_ufm128_404_sori"
# dataset_name = "verysmallset"
dataset_name = "tiny_extract_m404"
# model_name = "ufm"
model_name = "4 layer Transformer pos off"
# dataset_name = "tiny_extracted_m100"
tok_file = f'./data/{dataset_name}_pretok_word.bin'
record_dir = "/Users/yizezhao/Desktop/ExpPaperV2"

s_len = 3
# set_s = "equal"
# set_s = "random"
set_s = "original"
range_repeat = 5
# range_repeat = 3
# save_pretok = None

ufm_verysmall_result = f"{record_dir}/results_verysmallset_cpu_verysmall_ufm_128_fix_s3"
tfm_verysmall_result = f"{record_dir}/results_verysmallset_cpu_verysmall_tfm_128_fix_s3"
ufm_tiny404_result = f"{record_dir}/results_tiny_extract_m404_cpu_ufm128_404_sori"
tfm_tiny404_result = f"{record_dir}/results_tiny_extract_m404_colab_tfm128_404_sori"
if model_name == "ufm":
    if dataset_name == "verysmallset":
        result_dir = ufm_verysmall_result
    else:
        result_dir = ufm_tiny404_result
else :
    if dataset_name == "verysmallset":
        result_dir = tfm_verysmall_result
    else:
        result_dir = tfm_tiny404_result


tokenizer_file = f'./tokenizer/tok_{dataset_name}_word.model'
sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
vocab_size = sp.get_piece_size()

# result_dir = ufm_verysmall_result

figure_dir = f'{record_dir}/geometry_figures'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

bos = 1
eos = 2

# model
init_from = "tfm" # mlp, tfm or ufm

# mlp specific
d_encode = 128
d_hidden = 128
d_decode = 128

# tfm specific
dim = 128
n_layers = 4
n_heads = 6
n_kv_heads = 6
multiple_of = 4
dropout = 0.0
pos_enc = "off"
max_seq_len = 6

#ufm specific1
# dim = 512

# optimizer
lr = 1e-4
lammy = 1e-6
#training
max_iteration = 30000
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


#model_args = dict(T=T, v_ctx=vocab_size, v_nt=vocab_size, d_input=d_encode, d_hidden=d_hidden, d_output=d_decode)

# load back  W_list, L_list, H_list
# S = np.load(f'{result_dir}/S_{model_name}_d{dim}.npy')
P = np.load(f'{result_dir}/P_{model_name}_d{dim}.npy')
# imshow the P
second_sst = S@S.T@S@S.T@S@S.T
# plt.imshow(second_sst)
# plt.colorbar()
# plt.savefig(f'{figure_dir}/3ndSST_{model_name}_d{dim}.pdf')
# plt.show()
W_list = np.load(f'{result_dir}/W_list_{model_name}_d{dim}.npy')
L_list = np.load(f'{result_dir}/L_list_{model_name}_d{dim}.npy')
H_list = np.load(f'{result_dir}/H_list_{model_name}_d{dim}.npy')
losses = np.load(f'{result_dir}/losses_{model_name}_d{dim}.npy')
rec = np.load(f'{result_dir}/rec_{model_name}_d{dim}.npy')

# model.load_state_dict(torch.load(f'{result_dir}/model_{model.name}.pth', map_location=torch.device(device)))
# exit()
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

def plotloss404():
    tsk_entropy = 1.6597
    loss_list_ufm = np.load(f'{ufm_tiny404_result}/losses_UFM_d128.npy')
    loss_list_tfm = np.load(f'{tfm_tiny404_result}/losses_4 layer Transformer pos off_d128.npy')
    # plot two losses on the same plot
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    axs.plot(range(len(loss_list_ufm)), loss_list_ufm - tsk_entropy, label='CE loss - Entropy UFM')
    axs.plot(range(len(loss_list_tfm)), loss_list_tfm - tsk_entropy, label='CE loss - Entropy TFM')
    # axs[0].axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')

    # Add labels and legend
    axs.set_xlabel('Iteration')
    axs.set_ylabel('loss - entropy')
    axs.set_yscale('log')
    # axs.set_title(f'Cross Entropy loss with UFM and TFM \n on {dataset_name} \n H={tsk_entropy:.4f}')
    axs.legend()

    # Show the plot
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if not os.path.exists(f'{figure_dir}'):
        os.makedirs(f'{figure_dir}')
    plt.savefig(f'{figure_dir}/CEloss_UFM_TFM_tiny404.pdf')
    # plt.show()

def plotlossverysmall():

    loss_list_ufm = np.load(f'{ufm_verysmall_result}/losses_UFM_d128.npy')
    loss_list_tfm = np.load(f'{tfm_verysmall_result}/losses_4 layer Transformer pos off_d128.npy')
  # plot two losses on the same plot
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, dpi=100)

    axs.plot(range(len(loss_list_ufm)), loss_list_ufm - tsk_entropy, label='CE loss - Entropy UFM')
    axs.plot(range(len(loss_list_tfm)), loss_list_tfm - tsk_entropy, label='CE loss - Entropy TFM')
    # axs[0].axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')

    # Add labels and legend
    axs.set_xlabel('Iteration')
    axs.set_ylabel('loss - entropy')
    axs.set_yscale('log')
    # axs.set_title(f'Cross Entropy loss with UFM and TFM \n on {dataset_name} \n H={tsk_entropy:.4f}')
    axs.legend()

    # Show the plot
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if not os.path.exists(f'{figure_dir}'):
        os.makedirs(f'{figure_dir}')
    plt.savefig(f'{figure_dir}/CEloss_UFM_TFM_verysmallset.pdf')
    # plt.show()
# exit(0)
# SVM
# print(embeds)
# plotlossverysmall()

# plot_train()
# exit(0)
# W convergence





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


def W_H_norm_plot():
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, dpi=100)

    axs.plot(rec, W_norm, label='||W_t||')
    axs.plot(rec, H_norm, label='||H_t||')
    # axs[0].set_title('Norm')
    # axs.set_xlabel('iterations')
    axs.legend()

    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(f'{figure_dir}/WHnormgrowth_{dataset_name}_{model_name}_d{dim}.pdf')
    # plt.show()



# exit(0)

# Lmm, GW
# if init_from == "ufm" or init_from == "tfm":
S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
# if solve_by_cvx:
# L_mm_cvxpy = SVM_cvxpy_sol(S)
# load lmm cvxpy
if dataset_name == "verysmallset":
    L_mm_cvxpy = np.load(f'{ufm_verysmall_result}/Lmm_cvxpy_verysmallset.npy')
else:
    L_mm_cvxpy = np.load(f'{ufm_tiny404_result}/Lmm_cvxpy_{dataset_name}.npy')

L_mm_cvxpy_normed = L_mm_cvxpy / np.linalg.norm(L_mm_cvxpy)

# save lmm cvxpy
np.save(f'{result_dir}/Lmm_cvxpy_{dataset_name}.npy', L_mm_cvxpy)
# else:
L_mm_thm = SVM_thm_sol(tsk.ctx_dict, support_set_sampled, v_nt)
L_mm_thm_normed = L_mm_thm / np.linalg.norm(L_mm_thm)
# l_mm_normed = L_mm / np.linalg.norm(L_mm)

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

print("computing grams for Lmm cvxpy")
WG_mm_norm_cvxpy, HG_mm_norm_cvxpy, W_mm_cvxpy, H_mm_cvxpy, u_cvxpy, vh_cvxpy = compute_grams(L_mm_cvxpy, v_nt)
print("computing grams for Lmm thm")
WG_mm_norm_thm, HG_mm_norm_thm, W_mm_thm, H_mm_thm, u_thm, vh_thm = compute_grams(L_mm_thm, v_nt)


def plot_logits_verysmallsets():
    L_list_ufm = np.load(
        f'{ufm_verysmall_result}/L_list_UFM_d128.npy')
    L_list_tfm = np.load(
        f'{tfm_verysmall_result}/L_list_4 layer Transformer pos off_d128.npy')
    # rec = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_tiny_extract_m404_cpu_ufm128_404_sori/rec_UFM_d128.npy')
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
    # rec = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_tiny_extract_m404_cpu_ufm128_404_sori/rec_UFM_d128.npy')
    L_ufm = L_list_ufm[-1]
    L_ufm_normed = L_ufm / np.linalg.norm(L_ufm)
    L_tfm = L_list_tfm[-1]
    L_tfm_normed = L_tfm / np.linalg.norm(L_tfm)
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
    im3 = axs.imshow(Lmm_data_normed.T, cmap='hot', interpolation='nearest')
    # axs.set_title('L_mm_cvxpy')
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im3)
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
    S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
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


def plot_norms():

    nrows, ncols = 2,3
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    axs[0,0].plot(rec, W_norm, label='||W_t||')
    axs[0,0].plot(rec, H_norm, label='||H_t||')
    axs[0,0].set_title('Norm')
    axs[0,0].set_xlabel('epoch')
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
    axs[1, 0].plot(rec, HG_proj_diff_cvxpy, label="||GW_t_proj - GW_mm|| cvxpy", linestyle='--', color='red')
    axs[1, 0].set_title('||GW_t - GW_mm||')
    axs[1, 0].set_xlabel('epoch')
    axs[1, 0].legend()

    axs[1, 1].plot(rec, HG_diff_thm, label="||GH_t - GH_mm|| theorem")
    axs[1, 1].plot(rec, HG_diff_cvxpy, label="||GH_t - GH_mm|| cvxpy", linestyle='--', color='green')
    axs[1, 1].set_title('||GH_t - GH_mm||')
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
        if it == -1:
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

def nth_order_support_plots():
    # plot H.T@H, S.T@S, P.T@P, S.T@S@S.T@S, P.T@P@P.T@P
    H = H_list[-1]
    W = W_list[-1]

    plot_matrices_row1 = [H, S, P, S@S.T, P@P.T, S@S.T@S, P@P.T@P]
    plot_matrices_names_row1 = ["H", "S", "P", "2-nd S", "2-nd P", "3-rd S", "3-rd P"]

    plot_matrices_row2 = [W, S.T, P.T, S.T@S, P.T@P, S.T@S@S.T, P.T@P@P.T]
    plot_matrices_names_row2 = ["W", "S_T", "P_T", "2-nd S_T", "2-nd P_T", "3-rd S_T", "3-rd P_T"]

    nrows, ncols = 2, len(plot_matrices_row1)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    for j in range(len(plot_matrices_row1)):
        A1 = plot_matrices_row1[j]
        N1 = plot_matrices_names_row1[j]
        # A1mean = A1.mean(axis=1, keepdims=True)
        # A1 = A1 - A1mean
        AAT1 = A1@A1.T
        AAT_norm1 = AAT1/norm(AAT1)

        A2 = plot_matrices_row2[j]
        N2 = plot_matrices_names_row2[j]
        # A2mean = A2.mean(axis=1, keepdims=True)
        # A2 = A2 - A2mean
        AAT2 = A2@A2.T
        AAT_norm2 = AAT2/norm(AAT2)

        im1 = axs[0, j].imshow(AAT_norm1, cmap='viridis', interpolation='nearest')
        axs[0, j].set_title(f'{N1}{N1}.T')
        plt.colorbar(im1)


        im2 = axs[1, j].imshow(AAT_norm2, cmap='viridis', interpolation='nearest')
        axs[1, j].set_title(f'{N2}{N2}.T')
        plt.colorbar(im2)

    plt.savefig(f'{figure_dir}/nth_order_support_{model_name}_d{dim}.pdf')
    # plt.show()

def nth_order_support_sim_plots():
    # plot H.T@H, S.T@S, P.T@P, S.T@S@S.T@S, P.T@P@P.T@P
    H = H_list[-1]
    W = W_list[-1]
    L = L_list[-1]

    print(f"number of edges: {S.sum()}")

    plot_matrices_row1 = [H, S, P, S@S.T, P@P.T, S@S.T@S, P@P.T@P, L]
    plot_matrices_names_row1 = ["H", "S", "P", "2-nd S", "2-nd P", "3-rd S", "3-rd P", "L"]

    plot_matrices_row2 = [W, S.T, P.T, S.T@S, P.T@P, S.T@S@S.T, P.T@P@P.T, L.T]
    plot_matrices_names_row2 = ["W", "S_T", "P_T", "2-nd S_T", "2-nd P_T", "3-rd S_T", "3-rd P_T", "L_T"]

    nrows, ncols = 2, len(plot_matrices_row1)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    for j in range(len(plot_matrices_row1)):
        A1 = plot_matrices_row1[j]
        N1 = plot_matrices_names_row1[j]
        # AAT1 = A1@A1.T
        # A1mean = A1.mean(axis=1, keepdims=True)
        # A1 = A1 - A1mean
        Asim1 = pairwise_cossim(A1)
        # AAT_norm1 = AAT1/norm(AAT1)

        A2 = plot_matrices_row2[j]
        N2 = plot_matrices_names_row2[j]
        # AAT2 = A2@A2.T
        # AAT_norm2 = AAT2/norm(AAT2)
        # A2mean = A2.mean(axis=1, keepdims=True)
        # A2 = A2 - A2mean
        Asim2 = pairwise_cossim(A2)

        im1 = axs[0, j].imshow(Asim1, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
        axs[0, j].set_title(f'Cos({N1})')
        plt.colorbar(im1)


        im2 = axs[1, j].imshow(Asim2, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
        axs[1, j].set_title(f'Cos({N2})')
        plt.colorbar(im2)

    plt.savefig(f'{figure_dir}/nth_order_support_sim_{model_name}_d{dim}.pdf')
    # plt.show()

# nth_order_support_sim_plots()


def plot_geometry_cluster(cluster_def = "ball"):
    v_nt = S.shape[1]
    m = S.shape[0]
    H_unorm = H_list[-1]
    W_unorm = W_list[-1]

    avg_H_norm = np.linalg.norm(H_unorm, axis=1).mean()
    avg_W_norm = np.linalg.norm(W_unorm, axis=1).mean()
    H = H_unorm / avg_H_norm
    W = W_unorm / avg_W_norm
    print(H.shape)
    print(W.shape)

    H_ctx_means = np.zeros((v_nt, H.shape[1]))
    W_cts_means = np.zeros((m, W.shape[1]))

    z_not_satisfy = 0
    j_not_satisfy = 0

    for z in range(v_nt):
        ctx_idx = np.where(S[:,z] == 1)[0]
        H_ctx = H[ctx_idx,:]
        # check if the W[z,:] is in the cone of H_ctx
        H_ctx_means[z] = H_ctx.mean(axis=0)

        if len(ctx_idx) <=1:
            continue

        if cluster_def == "ball":
            ball = compute_ball_from_boundary(list(H_ctx))
            lies = is_in_ball(ball, W[z,:])
            if not lies:
                z_not_satisfy += 1
                print(f"Word {z} not in the ball of its context, ctx length={len(ctx_idx)}")

        if cluster_def == "cone":
            lies = lies_in_cone(H_ctx, W[z,:])
            # lies = lies_in_any_cone(H_ctx, W[z,:])
            if not lies:
                z_not_satisfy += 1
                print(f"Word {z} not in the cone of its context, ctx length={len(ctx_idx)}")

        # cube
        if cluster_def == "cube":
            cube = find_smallest_enclosing_cube(H_ctx)
            lies = is_in_cube(cube, W[z,:])
            if not lies:
                z_not_satisfy += 1
                print(f"Word {z} not in the cube of its context, ctx length={len(ctx_idx)}")


    for j in range(m):
        words_idx = np.where(S[j,:] >0)[0]
        W_idx = W[words_idx,:]

        W_cts_means[j] = W_idx.mean(axis=0)
        if len(words_idx) <=1:
            continue
        # cone
        if cluster_def == "cone":
            lies = lies_in_cone(W_idx, H[j,:])
            # lies = lies_in_any_cone(W_idx, H[j,:])
            if not lies:
                j_not_satisfy += 1
                print(f"Context {j} not in the cone of its words, ctx length={len(words_idx)}")
        # ball
        if cluster_def == "ball":
            ball = compute_ball_from_boundary(list(W_idx))
            lies = is_in_ball(ball, H[j,:])
            if not lies:
                j_not_satisfy += 1
                print(f"Context {j} not in the ball of its words, ctx length={len(words_idx)}")
        # cube
        if cluster_def == "cube":
            cube = find_smallest_enclosing_cube(W_idx)
            lies = is_in_cube(cube, H[j,:])
            if not lies:
                j_not_satisfy += 1
                print(f"Context {j} not in the cube of its words, ctx length={len(words_idx)}")
    print(f"{cluster_def} -- Words satisfy: {(v_nt - z_not_satisfy)/v_nt}, Contexts satisfy: {(m - j_not_satisfy)/m}")

    norms_H_ctx_means = np.linalg.norm(H_ctx_means, axis=1, keepdims=True)
    norms_W = np.linalg.norm(W.T, axis=0)

    norms_W_idx_means = np.linalg.norm(W_cts_means, axis=1, keepdims=True)
    norms_H = np.linalg.norm(H.T, axis=0)

    sim_H_W = H_ctx_means @ W.T / (norms_H_ctx_means * norms_W)
    sim_W_H = W_cts_means @ H.T / (norms_W_idx_means * norms_H)

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    simH_H = pairwise_cossim(H)
    simW_W = pairwise_cossim(W)

    # plot H.T@H and W.T@W
    im3 = axs[0,1].imshow(simH_H, cmap='viridis', interpolation='nearest')
    axs[0,1].set_title('H pairwise similarity')
    im4 = axs[0,0].imshow(simW_W, cmap='viridis', interpolation='nearest')
    axs[0,0].set_title('W pairwise similarity')


    im1 = axs[1,0].imshow(sim_H_W, cmap='viridis', interpolation='nearest')
    axs[1,0].set_title('H_ctx_means @ W.T')
    im2 = axs[1,1].imshow(sim_W_H, cmap='viridis', interpolation='nearest')
    axs[1,1].set_title('W_cts_means @ H.T')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.colorbar(im3)
    plt.colorbar(im4)

    plt.savefig(f'{figure_dir}/sim_H_W_mean{model_name}_d{dim}.pdf')
    print("HmeanW plot saved in ", f'{figure_dir}/sim_H_W_mean{model_name}_d{dim}.pdf')


def pairwise_euclidean_distance(H, W):
    # Compute the squared sum of each row in H and W
    H_sq = np.sum(H ** 2, axis=1).reshape(-1, 1)
    W_sq = np.sum(W ** 2, axis=1).reshape(1, -1)

    # Compute the pairwise Euclidean distances
    distances = np.sqrt(H_sq + W_sq - 2 * np.dot(H, W.T))
    return distances

def min_dist_satisfy():
    H_z_wz_dist_means = []
    W_j_hj_dist_means = []
    z_satisfy_rate = []
    j_satisfy_rate = []
    for e in range(0, len(H_list)):
        v_nt = S.shape[1]
        m = S.shape[0]
        H_unorm = H_list[e]
        W_unorm = W_list[e]

        avg_H_norm = np.linalg.norm(H_unorm, axis=1).mean()
        avg_W_norm = np.linalg.norm(W_unorm, axis=1).mean()
        H = H_unorm / avg_H_norm
        W = W_unorm / avg_W_norm
        print(H.shape)
        print(W.shape)

        H_ctx_means = np.zeros((v_nt, H.shape[1]))
        W_cts_means = np.zeros((m, W.shape[1]))

        H_z_wz_dists = []
        W_j_hj_dists = []

        z_satisfy_cnt = 0
        j_satisfy_cnt = 0
        for z in range(v_nt):
            ctx_idx = np.where(S[:, z] == 1)[0]
            H_ctx = H[ctx_idx, :]
            # check if the W[z,:] is in the cone of H_ctx
            # H_ctx_means[z] = H_ctx.mean(axis=0)
            # dists_to_z = np.linalg.norm(W - H_ctx_means[z], axis=1)
            # min_dist = np.min(dists_to_z)
            # min_dist_idx = np.argmin(dists_to_z)
            # compute pairwise distance between W and H_ctx
            dists_to_z = pairwise_euclidean_distance(H_ctx, W)
            avg_dists = np.mean(dists_to_z, axis=1)
            min_dist = np.min(avg_dists)
            min_dist_idx = np.argmin(avg_dists)


            if min_dist_idx == z:
                z_satisfy_cnt += 1
            H_z_wz_dists.append(np.linalg.norm(W[z, :] - H_ctx.mean(axis=0)))

        for j in range(m):
            words_idx = np.where(S[j, :] > 0)[0]
            W_idx = W[words_idx, :]
            W_cts_means[j] = W_idx.mean(axis=0)
            # dists_to_j = np.linalg.norm(H - W_cts_means[j], axis=1)
            # min_dist = np.min(dists_to_j)
            # min_dist_idx = np.argmin(dists_to_j)
            # compute pairwise distance between H and W_idx
            dists_to_j = pairwise_euclidean_distance(W_idx, H)
            avg_dists = np.mean(dists_to_j, axis=1)
            min_dist = np.min(avg_dists)
            min_dist_idx = np.argmin(avg_dists)

            if min_dist_idx == j:
                j_satisfy_cnt += 1
            # W_j_hj_dists.append(np.linalg.norm(H[j, :] - W_idx.mean(axis=0)))

        # H_z_wz_dist_means.append(np.mean(H_z_wz_dists))
        # W_j_hj_dist_means.append(np.mean(W_j_hj_dists))
        z_satisfy_rate.append(z_satisfy_cnt/v_nt)
        j_satisfy_rate.append(j_satisfy_cnt/m)

        # print(f"Words mean dist: {np.mean(H_z_wz_dists)}, Contexts mean dist: {np.mean(W_j_hj_dists)}")

    # plt.figure()
    # plt.plot(rec, H_z_wz_dist_means, label='H_z_wz_dist_means')
    # plt.plot(rec, W_j_hj_dist_means, label='W_j_hj_dist_means')
    # plt.title(f"Mean distance between H_z and W_z, H_j and W_j")
    # plt.legend()
    # plt.savefig(f'{figure_dir}/dist_mean_{model_name}_d{dim}.pdf')
    # print("Mean dist plot saved in ", f'{figure_dir}/dist_mean_{model_name}_d{dim}.pdf')

    plt.figure()
    plt.plot(rec, z_satisfy_rate, label='z_satisfy_rate')
    plt.plot(rec, j_satisfy_rate, label='j_satisfy_rate')
    plt.title(f"Rate of min dist satisfy")
    plt.legend()
    plt.savefig(f'{figure_dir}/min_dist_satisfy_{model_name}_d{dim}.pdf')

    print("min dist satisfy plot saved in ", f'{figure_dir}/min_dist_satisfy_{model_name}_d{dim}.pdf')

min_dist_satisfy()
# calc_geometry_cluster()


# plot_geometry_cluster()

def calc_geometry_cluster():
    H_z_wz_dist_means = []
    W_j_hj_dist_means = []
    for e in range(0, len(H_list)):
        v_nt = S.shape[1]
        m = S.shape[0]
        H_unorm = H_list[e]
        W_unorm = W_list[e]

        avg_H_norm = np.linalg.norm(H_unorm, axis=1).mean()
        avg_W_norm = np.linalg.norm(W_unorm, axis=1).mean()
        H = H_unorm / avg_H_norm
        W = W_unorm / avg_W_norm
        print(H.shape)
        print(W.shape)

        H_ctx_means = np.zeros((v_nt, H.shape[1]))
        W_cts_means = np.zeros((m, W.shape[1]))

        H_z_wz_dists = []
        W_j_hj_dists = []


        for z in range(v_nt):
            ctx_idx = np.where(S[:,z] == 1)[0]
            H_ctx = H[ctx_idx,:]
            # check if the W[z,:] is in the cone of H_ctx
            H_ctx_means[z] = H_ctx.mean(axis=0)

            if len(ctx_idx) <=1:
                continue
            H_z_wz_dists.append(np.linalg.norm(W[z,:] - H_ctx.mean(axis=0)))


        for j in range(m):
            words_idx = np.where(S[j,:] >0)[0]
            W_idx = W[words_idx,:]

            W_cts_means[j] = W_idx.mean(axis=0)
            if len(words_idx) <=1:
                continue
            W_j_hj_dists.append(np.linalg.norm(H[j,:] - W_idx.mean(axis=0)))

        H_z_wz_dist_means.append(np.mean(H_z_wz_dists))
        W_j_hj_dist_means.append(np.mean(W_j_hj_dists))
        print(f"Words mean dist: {np.mean(H_z_wz_dists)}, Contexts mean dist: {np.mean(W_j_hj_dists)}")


    plt.figure()
    plt.plot(rec, H_z_wz_dist_means, label='H_z_wz_dist_means')
    plt.plot(rec, W_j_hj_dist_means, label='W_j_hj_dist_means')
    plt.title(f"Mean distance between H_z and W_z, H_j and W_j")
    plt.legend()
    plt.savefig(f'{figure_dir}/dist_mean_{model_name}_d{dim}.pdf')
    print("Mean dist plot saved in ", f'{figure_dir}/dist_mean_{model_name}_d{dim}.pdf')
# calc_geometry_cluster()

def plot_geometry_HmeanW():
    v_nt = S.shape[1]
    m = S.shape[0]
    H = H_list[-1]
    W = W_list[-1]
    print(H.shape)
    print(W.shape)

    H_ctx_means = np.zeros((v_nt, H.shape[1]))
    W_cts_means = np.zeros((m, W.shape[1]))

    # find the most frequent word
    word_freq = S.sum(axis=0)
    word_freq = word_freq / word_freq.sum()
    word_freqs_argsort = word_freq.argsort()[::-1]

    most, least, mid = word_freqs_argsort[0], word_freqs_argsort[-1], word_freqs_argsort[len(word_freqs_argsort)//2]

    print(most, least, mid)
    H_to_plot = []
    ctx_len = []

    for z in list([most, least, mid]):
        ctx_idx = np.where(S[:,z] == 1)[0]
        H_sel = H[ctx_idx,:]
        H_not_sel = H[np.where(S[:,z] == 0)[0],:]
        H_new = np.vstack([H_sel, H_not_sel])
        # re-order the H_ctx so H_cts are sorted first
        H_to_plot.append(H_new)
        ctx_len.append(len(ctx_idx))

    nrows, ncols = 1, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    print(len(H_to_plot))

    for z in range(len(H_to_plot)):
        print(z)
        H = H_to_plot[z]
        simH_H = pairwise_cossim(H)
        im1 = axs[z].imshow(simH_H, cmap='viridis', interpolation='nearest')
        axs[z].set_title(f"H pairwise similarity {['most', 'least', 'mid'][z]}, ctx_len={ctx_len[z]}")
        plt.colorbar(im1)
    plt.savefig(f'{figure_dir}/sim_H_W_cluster{model_name}_d{dim}.pdf')
    print("cluster plot saved in ", f'{figure_dir}/sim_H_W_cluster{model_name}_d{dim}.pdf')

# plot_geometry_HmeanW()
def plot_cluster_euclidean():
    v_nt = S.shape[1]
    m = S.shape[0]
    H = H_list[-1]
    W = W_list[-1]
    print(H.shape)
    print(W.shape)


    # find the most frequent word
    word_freq = S.sum(axis=0)
    word_freq = word_freq / word_freq.sum()
    word_freqs_argsort = list(word_freq.argsort()[::-1])

    dist2mean_euc = []

    for z in (word_freqs_argsort):
        ctx_idx = np.where(S[:,z] == 1)[0]
        H_sel = H[ctx_idx,:]
        # Hz_mean = H_sel.mean(axis=0)
        Hz_euclideans = np.mean(np.linalg.norm(H_sel - H_sel.mean(axis=0), axis=1))
        dist2mean_euc.append(Hz_euclideans)

    plt.figure()
    plt.plot(dist2mean_euc, label='euc_dist to mean', marker='o')
    plt.title(f"Euclidean distance to the mean of H_z")
    plt.savefig(f'{figure_dir}/dist_euc_{model_name}_d{dim}.pdf')
    print("Euclidean dist plot saved in ", f'{figure_dir}/dist_euc_{model_name}_d{dim}.pdf')





# plot_cluster_euclidean()


def plot_cluster_covanriance():
    v_nt = S.shape[1]
    m = S.shape[0]
    H = H_list[-1]
    W = W_list[-1]
    print(H.shape)
    print(W.shape)

    # H_ctx_means = np.zeros((v_nt, H.shape[1]))
    # W_cts_means = np.zeros((m, W.shape[1]))

    # find the most frequent word
    word_freq = S.sum(axis=0)
    word_freq = word_freq / word_freq.sum()
    word_freqs_argsort = list(word_freq.argsort()[::-1])

    # most, least, mid = word_freqs_argsort[0], word_freqs_argsort[-1], word_freqs_argsort[len(word_freqs_argsort)//2]

    # print(most, least, mid)
    cov_trace = []

    for z in (word_freqs_argsort):
        ctx_idx = np.where(S[:,z] == 1)[0]
        H_sel = H[ctx_idx,:]
        # Hz_mean = H_sel.mean(axis=0)
        Hz_cov = np.cov(H_sel.T, rowvar=True, bias=True)
        # print(f"cov shape: {Hz_cov.shape}")
        cov_trace.append(np.trace(Hz_cov))

    plt.figure()
    plt.plot(cov_trace, label='cov trace', marker='o')
    plt.title(f"trace of the covariance matrix of H_z")
    plt.savefig(f'{figure_dir}/cov_trace_{model_name}_d{dim}.pdf')
    print("cov trace plot saved in ", f'{figure_dir}/cov_trace_{model_name}_d{dim}.pdf')

# plot_cluster_covanriance()


def find_column_pairs(matrix):
    # This list will hold the pairs of column indices
    column_pairs = []

    # Iterate over each row in the matrix
    for row in matrix:
        # Find indices where the row elements are 1
        indices = np.where(row == 1)[0]
        # Generate all combinations of the indices in pairs
        for combo in combinations(indices, 2):
            # Append each pair to the list (sorted to avoid duplicates like (2,1) and (1,2))
            column_pairs.append(tuple(sorted(combo)))

    # Remove duplicates by converting the list to a set, then back to a list
    unique_pairs = list(set(column_pairs))
    return unique_pairs

def plot_mult_2_structure():
    # find all word pairs that appeared in at least 2 contexts
    v_nt = S.shape[1]
    m = S.shape[0]
    H = H_list[-1]
    W = W_list[-1]
    print(H.shape)
    print(W.shape)

    word_freq = S.sum(axis=0)

    word_pairs = find_column_pairs(S)
    print(len(word_pairs))

    # for each word paor, find the contexts that contains both words
    pair_ctxs_diff = []
    for pair in word_pairs:
        ctx1 = np.where(S[:,pair[0]] == 1)[0]
        weight1 = len(ctx1)
        H_z1 = H[ctx1,:]
        ctx2 = np.where(S[:,pair[1]] == 1)[0]
        weight2 = len(ctx2)
        H_z2 = H[ctx2,:]
        ctx_both = np.intersect1d(ctx1, ctx2)
        H_z12 = H[ctx_both,:]
        # pair_ctxs.append(ctx_both)
        weight12 = weight1 + weight2
        # find the difference between the average between the two contexts and the context that contains both
        H_z12_weighted = (weight1 * H_z1.mean(axis=0) + weight2 * H_z2.mean(axis=0))/weight12
        H_z12_diff = np.linalg.norm(H_z12.mean(axis=0) - H_z12_weighted)
        pair_ctxs_diff.append(H_z12_diff)

        # check if each of H_z12

    # plot the difference in the average of the two contexts and the context that contains both
    plt.figure()
    plt.plot(pair_ctxs_diff, label="H_z12_center - H_z1_H_z2_weighted", marker='o')
    plt.title(f"Euclidean distance to the weighted average of the two contexts")
    plt.legend()
    plt.savefig(f'{figure_dir}/pair_diff_{model_name}_d{dim}.pdf')
    print("pair diff plot saved in ", f'{figure_dir}/pair_diff_{model_name}_d{dim}.pdf')

# plot_mult_2_structure()

def check_nc1():
    v_nt = S.shape[1]
    m = S.shape[0]

    nc1_means = []
    for e in range(0, len(H_list)):
        H_unorm = H_list[e]
        W_unorm = W_list[e]

        avg_H_norm = np.linalg.norm(H_unorm, axis=1).mean()
        avg_W_norm = np.linalg.norm(W_unorm, axis=1).mean()
        H = H_unorm / avg_H_norm
        W = W_unorm / avg_W_norm
        print(H.shape)
        print(W.shape)


        # find all diffferent rows in S
        S_unique = np.unique(S, axis=0)
        print(f"number of unique rows in S: {S_unique.shape[0]}")

        # most, least, mid = word_freqs_argsort[0], word_freqs_argsort[-1], word_freqs_argsort[len(word_freqs_argsort)//2]

        # print(most, least, mid)
        cov_trace = []

        for s_ in (S_unique):
            ctx_idx = np.where(np.all(S == s_, axis=1))[0]
            H_sel = H[ctx_idx,:]
            # Hz_mean = H_sel.mean(axis=0)
            Hz_cov = np.cov(H_sel.T, rowvar=True, bias=True)
            # print(f"cov shape: {Hz_cov.shape}")
            cov_trace.append(np.trace(Hz_cov))

        nc1_means.append(np.mean(cov_trace))

    plt.figure()
    plt.plot(rec, nc1_means, label='cov trace means')
    plt.title(f"trace of the covariance matrix of H_z")
    plt.legend()
    plt.savefig(f'{figure_dir}/nc1_{model_name}_d{dim}.pdf')
    print("nc1 plot saved in ", f'{figure_dir}/nc1_s_{model_name}_d{dim}.pdf')
    # plt.figure()
    # plt.plot(cov_trace, label='cov trace', marker='o')
    # plt.title(f"trace of the covariance matrix of H_z")
    # plt.savefig(f'{figure_dir}/nc1_{model_name}_d{dim}.pdf')
    # print("nc1 plot saved in ", f'{figure_dir}/nc1_s_{model_name}_d{dim}.pdf')


check_nc1()
# # generate plot with Lmm cvxpy
# lmm_name="cvxpy"
# matrix_plot(WG_mm_norm_cvxpy, HG_mm_norm_cvxpy,L_mm_cvxpy_normed,lmm_name)
# embedding_similarity_plot(H_mm_cvxpy, lmm_name)
# matrix_plot_all(H_mm_cvxpy, W_mm_cvxpy,L_mm_cvxpy, "cvxpy")
# Sim_plot_all(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, "cvxpy")
# Sim_plot_iter(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, "cvxpy")
#
# # generate plot with Lmm thm
# lmm_name="theorem"
# matrix_plot(WG_mm_norm_thm, HG_mm_norm_thm,L_mm_thm_normed,lmm_name)
# embedding_similarity_plot(H_mm_thm, lmm_name)
# matrix_plot_all(H_mm_thm, W_mm_thm,L_mm_thm, lmm_name)
# Sim_plot_all(H_mm_thm, W_mm_thm, L_mm_thm, lmm_name)
# nth_order_support_plots()
# nth_order_support_sim_plots()




# H = np.array([
#     [2, 3],
#     [1, 2],
#     [3, 1]
# ])
# w = np.array([5, 8])
#
# print(lies_in_any_cone(H.T, w))




