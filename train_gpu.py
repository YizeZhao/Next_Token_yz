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
retrain = True
solve_by_cvx = False
check_lmm = True
# random.seed(1634)
# np.random.seed(1634)

# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "float32"  # float32|float16|float16

# dataset
batch_size = 32
if_batch=True
T = 6

dataset_name = "tiny_extract_m404"
# traininf_tag = "_cpu_verysmall_ufmtfm_128_fix_s3"
# traininf_tag = "mlp_12_layers_colab0523_run3"
traininf_tag = "tf_colab0525_d64_run1"

# dataset_name = "verysmallset"
# dataset_name = "tiny_extracted_m100"
tok_file = f'./data/{dataset_name}_pretok_word.bin'
# record_dir = "/Users/yizezhao/Documents/Models/NextToken"
record_dir = "./colab_results"
# record_dir = "/Users/yizezhao/Desktop/ExpPaperV1"
s_len = 3
# set_s = "equal"
# set_s = "random"
set_s = "original"
# range_repeat = 3
range_repeat = 5
# save_pretok = None


# model
init_from = "mlp" # mlp, tfm or ufm


tokenizer_file = f'./tokenizer/tok_{dataset_name}_word.model'
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
d_hiddens = [512] * 4 + [256] * 4 + [128] * 4
# print(d_hiddens)

# tfm specific
d_encode = 64
# d_hidden = 128
d_decode = 64
n_layers = 4
n_heads = 6
n_kv_heads = 6
multiple_of = 4
dropout = 0.0
pos_enc = "off"
max_seq_len = 6

#ufm specific1
# dim = 512
dim = 64

# optimizer
lr = 1e-4
lammy = 1e-6
#training
max_iteration = 10000
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
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 10000,15000,20000,25000], gamma=0.4)
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
if retrain:
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

        # if (iter == int(max_iteration*0.5)):
        #     # optimizer = torch.optim.Adam(model.parameters(), lr=lr/5, weight_decay=lammy)
        #     #optimizer = torch.optim.SGD(model.parameters(), lr=lr/5, weight_decay=lammy)
        #     # if init_from == "tfm":
        #
        #     for param in model.parameters():
        #         param.requires_grad = False
        #     for param in model.output.parameters\
        #                 ():
        #         param.requires_grad = True
        #     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
        #                                 lr=lr/5, weight_decay=lammy)

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
                if debug:
                    h_debug, _ = model(X, Y, return_embed=True)
                    l_debug = h_debug @ W_list[-1].T
                    loss_debug.append(F.cross_entropy(logits, Y).cpu().detach().numpy())

                if proj_each:
                    if init_from == "ufm":
                        uniq_embeds_dict = model.get_embeddings(tsk.ctx_dict)
                    else:
                        uniq_embeds_dict = model.get_embeddings(uniq_emb, tsk.v_ctx2v, tsk.ctx_dict)

                    w_fin, w_star = compute_svm(init_from, uniq_embeds_dict, support_set_sampled, support_set_pr,
                                                model_decode_d, v_nt)
                    if w_fin is None or w_star is None:
                        W_list.pop()
                        L_list.pop()
                        H_list.pop()
                        continue
                    # w_fin_normed = w_fin / np.linalg.norm(w_fin)
                    # w_star_normed = w_star / np.linalg.norm(w_star)
                    W_fin_list.append(w_fin)
                    W_star_list.append(w_star)
                    Smat = find_Smat(uniq_embeds_dict, support_set_sampled, s_len, v_nt, test_smat=False)
                    W_f.append(proj_f_w(Smat, W_list[-1]))

                    proj_rec.append(iter_)
                    #
                    # L_mm = SVM_thm_sol(tsk.ctx_dict, support_set_sampled, v_nt)
                    # l_mm_normed = L_mm / np.linalg.norm(L_mm)
                    #
                    # WG_mm_norm, HG_mm_norm = compute_grams(L_mm, v_nt)
                    # ww = W_list[-1]
                    # hh = H_list[-1]
                    #
                    # WG = ww @ ww.T
                    # HG = hh @ hh.T
                    #
                    # WG_list.append([WG])
                    # HG_list.append([HG])

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
    if debug:
        axs[0].plot(rec, loss_debug - tsk_entropy, label='debug_loss')
    # axs[0].axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')

    # Add labels and legend
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('loss')
    axs[0].set_yscale('log')
    axs[0].set_title(f'Cross Entropy loss with {model.name} \n on {dataset_name} \n H={tsk_entropy:.4f}')
    axs[0].legend()

    if check_wmm:
        axs[1].plot(plot_x, svm_losses - tsk_entropy, label='CE loss')
        # axs[1].axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')
        axs[1].set_xscale('log')
        axs[1].set_title('W_fin + alpha*W_star')
        axs[1].set_xlabel('alpha')
        axs[1].set_ylabel('loss')
        axs[1].set_yscale('log')
        axs[1].legend()


    # Show the plot
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if not os.path.exists(f'{figure_dir}'):
        os.makedirs(f'{figure_dir}')
    plt.savefig(f'{figure_dir}/CEloss{model.name}s_'+ str(s_len)+'.pdf')
    # plt.show()

# def plot_verysmall():
#     L_list_ufm = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_verysmallset_cpu_verysmall_ufm_128_fixed_s3/L_list_UFM.npy')
#     L_list_tfm = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_verysmallset_cpu_verysmall_tfm_128_fixed_s3/L_list_TFM.npy')

def plotloss404():


    L_list_ufm = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_tiny_extract_m404_cpu_ufm128_404_sori/losses_UFM_d128.npy')
    L_list_tfm = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_tiny_extract_m404_colab_tf128_404_sori/losses_8 layer Transformer pos off_d128.npy')
    # rec = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_tiny_extract_m404_cpu_ufm128_404_sori/rec_UFM_d128.npy')
    # rec_tfm = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_tiny_extract_m404_colab_tf128_404_sori/rec_8 layer Transformer pos off_d128.npy')
    # plot two losses on the same plot
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    axs.plot(range(len(L_list_ufm)), L_list_ufm - tsk_entropy, label='CE loss - Entropy UFM')
    axs.plot(range(len(L_list_tfm)), L_list_tfm - tsk_entropy, label='CE loss - Entropy TFM')
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
    plt.savefig(f'{figure_dir}/CEloss_UFM_TFM_{dataset_name}.pdf')
    # plt.show()

def plotlossverysmall():


    L_list_ufm = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_verysmallset_cpu_verysmall_ufmtfm_128_fix_s3/losses_UFM_d128.npy')
    L_list_tfm = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_verysmallset_cpu_verysmall_tfm_128_fixed/losses_4 layer Transformer pos off_d128.npy')
  # plot two losses on the same plot
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)

    axs.plot(range(len(L_list_ufm)), L_list_ufm - tsk_entropy, label='CE loss - Entropy UFM')
    axs.plot(range(len(L_list_tfm)), L_list_tfm - tsk_entropy, label='CE loss - Entropy TFM')
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
    plt.savefig(f'{figure_dir}/CEloss_UFM_TFM_{dataset_name}.pdf')
    # plt.show()

# plotloss404()
# plotlossverysmall()
# exit(0)

# SVM
# print(embeds)

if check_wmm:
    w_fin, w_star = compute_svm(init_from, uniq_embeds_dict, support_set_sampled, support_set_pr, model_decode_d, v_nt)
    # w_fin_normed = w_fin / np.linalg.norm(w_fin)
    # w_star_normed = w_star / np.linalg.norm(w_star)
    # set W = W_fin+infW* and compute entropy
    svm_losses, plot_x  = [], []
    embeds = model.forward_embedding(X)
    # for i in np.append(np.array([0,1]), (np.logspace(0.1, 3, 20, endpoint=True))):
    for i in np.logspace(0.1, 3, 20, endpoint=True):
        W = torch.Tensor(w_fin + i * w_star)
        logits = embeds@W.T
        ce_loss = F.cross_entropy(logits, Y)
        print(f"alpha={i} | loss {ce_loss:.4f} | entropy {tsk_entropy:.2f} ")
        svm_losses.append(ce_loss.detach().numpy())
        plot_x.append(i)

plot_train()



# W convergence
# W star direction convergence
#     check
#     1. Proj_f(W) -> W_fin
#     2. W->W_star
if do_svm:
    Smat = find_Smat(uniq_embeds_dict, support_set_sampled, s_len, v_nt, test_smat=False)
# exit()

    if proj_each:
        W_f = np.array(W_f)
        w_fin = np.array(W_fin_list)
        w_star = np.array(W_fin_list)
        WG_list = np.array(WG_list)
        HG_list = np.array(HG_list)
    else:
        W_f = np.zeros_like(W_list)
        for i in range(W_list.shape[0]):
            # print(i)
            W_f[i] = proj_f_w(Smat, W_list[i])

L_list_normed = L_list/(np.linalg.norm(L_list, axis=(1,2) ))[:, np.newaxis, np.newaxis]
W_list_normed = W_list/(np.linalg.norm(W_list, axis=(1,2) ))[:, np.newaxis, np.newaxis]
H_list_normed = H_list/(np.linalg.norm(H_list, axis=(1,2) ))[:, np.newaxis, np.newaxis]
W_dir = W_list_normed[1:] - W_list_normed[:-1]
L_dir = L_list_normed[1:] - L_list_normed[:-1]
H_dir = H_list_normed[1:] - H_list_normed[:-1]

if do_svm:
    Wf_Wfin_diff = compute_norm_corr(W_f, w_fin)
    Wdir_Wstar_diff = compute_norm_corr(W_dir, w_star)
    W_Wstar_diff = compute_norm_corr(W_list, w_star)
W_norm = np.linalg.norm(W_list, axis=(1,2) )
H_norm = np.linalg.norm(H_list, axis=(1,2) )
W_dir_norm = np.linalg.norm(W_dir, axis=(1,2) )
L_dir_norm = np.linalg.norm(L_dir, axis=(1,2))
H_dir_norm = np.linalg.norm(H_dir, axis=(1,2))




# Lmm, GW
# if init_from == "ufm" or init_from == "tfm":
S = get_s(tsk.ctx_dict, support_set_sampled, v_nt)
# if solve_by_cvx:
L_mm_cvxpy = SVM_cvxpy_sol(S)
L_mm_cvxpy_normed = L_mm_cvxpy / np.linalg.norm(L_mm_cvxpy)
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
WG_mm_norm_cvxpy, HG_mm_norm_cvxpy, W_mm_cvxpy, H_mm_cvxpy,_, _, WG_mm, HG_mm = compute_grams(L_mm_cvxpy, v_nt)
print("computing grams for Lmm thm")
WG_mm_norm_thm, HG_mm_norm_thm, W_mm_thm, H_mm_thm,_, _,  WG_mm, HG_mm = compute_grams(L_mm_thm, v_nt)


def plot_logits_404():
    L_list_ufm = np.load(
        f'/Users/yizezhao/Desktop/ExpPaperV1/results_tiny_extract_m404_cpu_ufm128_404_sori/L_UFM_d128.npy')
    L_list_tfm = np.load(
        f'/Users/yizezhao/Desktop/ExpPaperV1/results_tiny_extract_m404_colab_tf128_404_sori/L_8 layer Transformer pos off_d128.npy')
    # rec = np.load(f'/Users/yizezhao/Desktop/ExpPaperV1/results_tiny_extract_m404_cpu_ufm128_404_sori/rec_UFM_d128.npy')
    L_ufm = L_list_ufm[-1]
    L_tfm = L_list_tfm[-1]
    # imshow L_ufm, L_tfm and L_mm_cvxpy on the same colorbar
    color_max = np.max([np.max(L_ufm), np.max(L_tfm), np.max(L_mm_cvxpy)])
    color_min = np.min([np.min(L_ufm), np.min(L_tfm), np.min(L_mm_cvxpy)])

    # generate and save separate plots
    nrows, ncols = 1, 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    im1 = axs.imshow(L_ufm.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)

    axs.set_title('L_ufm')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im1)
    plt.savefig(f'{figure_dir}/L_ufm_{model.name}_d{dim}.pdf')

    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    im2 = axs.imshow(L_tfm.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)
    axs.set_title('L_tfm')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im2)

    plt.savefig(f'{figure_dir}/L_tfm_{model.name}_d{dim}.pdf')

    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    im3 = axs.imshow(L_mm_cvxpy.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)
    axs.set_title('L_mm_cvxpy')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im3)
    plt.savefig(f'{figure_dir}/L_mm_cvxpy_{model.name}_d{dim}.pdf')
    # plt.show()

    # generate and save a plot with all 3 matrices on the same colorbar
    nrows, ncols = 1, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    im1 = axs[0].imshow(L_ufm.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)
    axs[0].set_title('L_ufm')
    im2 = axs[1].imshow(L_tfm.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)
    axs[1].set_title('L_tfm')
    im3 = axs[2].imshow(L_mm_cvxpy.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)
    axs[2].set_title('L_mm_cvxpy')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.colorbar(im3)
    plt.savefig(f'{figure_dir}/L_ufm_tfm_mm_{model.name}_d{dim}.pdf')

# plot_logits_404()
# exit(0)


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
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    im1 = axs[0].imshow(L_mm_cvxpy.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)
    axs[0].set_title('L_mm_cvxpy')
    im2 = axs[1].imshow(L_mm_thm.T, cmap='hot', interpolation='nearest', vmax=color_max, vmin=color_min)
    axs[1].set_title('L_mm_thm')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.savefig(f'{figure_dir}/L_mm_cvxpy_thm_{model.name}_d{dim}.pdf')
    # plt.show()

    # exit(0)

HG_diff_cvxpy = np.zeros((len(W_list),1))
WG_diff_cvxpy = np.zeros((len(W_list),1))
HG_diff_thm = np.zeros((len(W_list),1))
WG_diff_thm = np.zeros((len(W_list),1))

for j in range(len(W_list)):
    ww = W_list[j]
    hh = H_list[j]

    WG = ww @ ww.T
    HG = hh @ hh.T

    WG_norm = WG / norm(WG,'fro')
    HG_norm = HG / norm(HG,'fro')

    wdiff_cvxpy = WG_norm - WG_mm_norm_cvxpy
    WG_diff_cvxpy[j] = norm(wdiff_cvxpy, 'fro')

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

    if do_svm:
        if proj_each:
            axs[0, 1].plot(proj_rec, Wf_Wfin_diff)
        else:
            axs[0, 1].plot(rec, Wf_Wfin_diff)
        axs[0, 1].set_title('||W_proj - W_fin||')
        axs[0, 1].set_xlabel('epoch')
        axs[1,0].plot(rec, W_Wstar_diff)
        axs[1,0].set_title('||W_t - W_star||')
        axs[1,0].set_xlabel('epoch')

        axs[1,1].plot(rec[1:], Wdir_Wstar_diff)
        axs[1,1].set_title('||W_update - W_star||')
        axs[1,1].set_xlabel('epoch')

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
    plt.savefig(f'{figure_dir}/Wnorm_{model.name}_d{dim}.pdf')
    # plt.show()


plot_norms()

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
    plt.savefig(f'{figure_dir}/Conv_{model.name}_d{dim}.pdf')
    # plt.show()

check_convergence_plot()

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
    plt.savefig(f'{figure_dir}/Matrices_{model.name}_d{dim}_{lmm_name}.pdf')
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

    plt.savefig(f'{figure_dir}/Matrices_all_{model.name}_d{dim}_{lmm_name}.pdf')
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
    ims = []
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

    plt.savefig(f'{figure_dir}/cossim_all_{model.name}_d{dim}_{lmm_name}.pdf')
    # plt.show()


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
    plt.savefig(f'{figure_dir}/Suppot_sim_{model.name}_d{dim}_{lmm_name}.pdf')
    # plt.show()



# generate plot with Lmm cvxpy
lmm_name="cvxpy"
matrix_plot(WG_mm_norm_cvxpy, HG_mm_norm_cvxpy,L_mm_cvxpy_normed,lmm_name)
embedding_similarity_plot(H_mm_cvxpy, lmm_name)
matrix_plot_all(H_mm_cvxpy, W_mm_cvxpy,L_mm_cvxpy, lmm_name)
Sim_plot_all(H_mm_cvxpy, W_mm_cvxpy, L_mm_cvxpy, lmm_name)

# generate plot with Lmm thm
lmm_name="theorem"
matrix_plot(WG_mm_norm_thm, HG_mm_norm_thm,L_mm_thm_normed,lmm_name)
embedding_similarity_plot(H_mm_thm, lmm_name)
matrix_plot_all(H_mm_thm, W_mm_thm,L_mm_thm, lmm_name)
Sim_plot_all(H_mm_thm, W_mm_thm, L_mm_thm, lmm_name)


exit(0)


def emb_geometry():
    # embedding geometry
    #sim_ctx = [[b'\x03\x00\x0e\x00\x04\x00&\x00\x06\x00' , b'\x03\x00\x0e\x00\x04\x00!\x00\x06\x00'], [b'\x03\x00\x18\x00\x04\x00\n\x00\x05\x00' ,  b'\x03\x00\x08\x00\x04\x00\x1b\x00\x05\x00'], [b'\x03\x00%\x00\x04\x00\x1a\x00\x06\x00' , b'\x03\x00\x08\x00\x04\x00\x13\x00\x06\x00']]
    sim_ctx = [[b'\x03\x00\x0e\x00\x04\x00&\x00\x06\x00', b'\x03\x00\x0e\x00\x04\x00!\x00\x06\x00', b'\x03\x00\t\x00\x0b\x00\x0f\x00\x05\x00', b"\x03\x00\x12\x00$\x00'\x00\x0b\x00", b'\x03\x00\x08\x00\x04\x00\x1d\x00\x11\x00', b'\x03\x00%\x00\x04\x00\x1a\x00\x06\x00', b'\x03\x00\x08\x00\x04\x00\x13\x00\x06\x00', b'\x03\x00\x08\x00\x04\x00"\x00\x05\x00', b'\x03\x00\x18\x00\x04\x00\n\x00\x05\x00', b'\x03\x00\x08\x00\x04\x00\x1b\x00\x05\x00', b'\x03\x00 \x00#\x00\x0f\x00\x05\x00'],\
                [b'\x03\x00\x17\x00\x10\x00\x07\x00\x0c\x00', b'\x03\x00\x16\x00\x14\x00\x07\x00\x0c\x00'],[b'\x06\x00\x19\x00\x15\x00\t\x00\x1f\x00',  b'\x03\x00\x1c\x00\n\x00\x07\x00\r\x00', b'\x03\x00\t\x00\x1e\x00\x07\x00\r\x00']]
    # # for k1, k2 in sim_ctx:
    #     sim_dist = np.linalg.norm(uniq_embeds_dict[k1] - uniq_embeds_dict[k2])
    #     diff_dists = []
    #     for k3, h3 in uniq_embeds_dict.items():
    #         if k3 == k1 or k3 == k2:
    #             continue
    #         diff_dists.append(np.linalg.norm(uniq_embeds_dict[k1] - uniq_embeds_dict[k3]))
    #         diff_dists.append(np.linalg.norm(uniq_embeds_dict[k2] - uniq_embeds_dict[k3]))
    #     diff_dist = np.array(diff_dists).mean()
    #     print(f"{k1+k2} | sim_dist {sim_dist:.4f} | diff_dist {diff_dist:.2f} | rate = {sim_dist/diff_dist:.2f}")
    cls_num = len(sim_ctx)
    for cls in range(cls_num):
        inter_dists, intra_dists = [], []
        for cls_ctx in sim_ctx[cls]:
            for cls_ in range(cls_num):
                for cls_ctx_ in sim_ctx[cls_]:
                    if cls == cls_:
                        intra_dists.append(np.linalg.norm(uniq_embeds_dict[cls_ctx] - uniq_embeds_dict[cls_ctx_]))
                    else:
                        inter_dists.append(np.linalg.norm(uniq_embeds_dict[cls_ctx] - uniq_embeds_dict[cls_ctx_]))
        inter_dists_mean = np.array(inter_dists).mean()
        intra_dists_mean = np.array(intra_dists).mean()
        print(len(inter_dists), len(intra_dists))
        print(f"class {cls} | sim_dist {intra_dists_mean:.4f} | diff_dist {inter_dists_mean:.4f} | rate = {intra_dists_mean / inter_dists_mean:.2f}")

emb_geometry()





