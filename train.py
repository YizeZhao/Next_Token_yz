'''
supports training of
1. fixed length next token
2. AR next token
3. on a relatively small dataset
'''

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt

from Dataset import *

import torch
from Model import Transformer, FixedLengthModelArgs, FixedLengthMLP, MultiLabelSVM, TFMModelArgs
import torch.nn.functional as F


# system
device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "float32"  # float32|bfloat16|float16

# dataset
batch_size = None
T = 6

tok_file = './data/verysmallset_pretok.bin'
s_len = 3
vocab_size = 152
tokenizer_file = './tokenizer/tok'+str(vocab_size)+'.model'
bos = 1
eos = 2

# model
init_from = "tfm"

# mlp specific
d_encode = 64
d_hidden = 128
d_decode = 64

# tfm specific
dim = 256
n_layers = 2
n_heads = 6
n_kv_heads = 6
multiple_of = 4
dropout = 0.0
pos_enc = "off"
max_seq_len = 6

# optimizer
lr = 5e-5
lammy = 0.5

#training
max_iteration = 6000

tsk = Task(batch_size=batch_size,
    T=T, tok_file=tok_file, s_len=s_len, vocab_size=vocab_size, bos=bos, eos=eos,
    device=device,)
# iter_batches = partial(
#     Task.iter_batches, batch_size=batch_size,
#     T=T, tok_file=tok_file, s_len=s_len, vocab_size=vocab_size, bos=bos, eos=eos,
#     device=device,)

#train_batch_iter = iter_batches()
tsk_entropy = tsk.emp_entropy
v_ctx = tsk.v_ctx
v_nt = tsk.v_nt
train_batch_iter = tsk.iter_batches()
X, Y = next(train_batch_iter)
print(X.shape, Y.shape)


#model_args = dict(T=T, v_ctx=vocab_size, v_nt=vocab_size, d_input=d_encode, d_hidden=d_hidden, d_output=d_decode)

if init_from == "mlp":
    # init a new model from scratch
    print("Initializing a new fixed MLP model from scratch")
    #fixedTconf = FixedLengthModelArgs(**model_args)
    model = FixedLengthMLP(T=T, v_ctx=v_ctx, v_nt=v_nt, d_input=d_encode, d_hidden=d_hidden, d_output=d_decode)

elif init_from == "tfm":
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


model.to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=lammy)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lammy)

# optimizer = torch.optim.Adam(model.parameters())


losses = []
W_norm = []
for iter in range(max_iteration):
    X, Y = next(train_batch_iter)

    # forward + backward + optimize
    if (iter+1)%100==0:
        # save normalized W
        print('debugging')
    logits = model(X,Y)
    loss = model.last_loss
    print(f"{iter} | loss {loss:.4f} | entropy {tsk_entropy:.2f} ")
    if loss < 0:
        break
    loss.backward()
    optimizer.step()
    losses.append(loss.detach().numpy())
    W_norm.append(np.linalg.norm\
                      (model.output.weight.detach().numpy()))





# losses = losses
plt.plot(np.arange(len(losses)), losses, label='CE loss')
plt.axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')

# Add labels and legend
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Cross Entropy loss with Fixed Length MLP on verysmalldataset S='+ str(s_len))
plt.legend()

# Show the plot
plt.savefig('./figures/FixedLengthMLPs_'+ str(s_len)+'.png')
plt.show()

plt.clf()
plt.plot(np.arange(len(W_norm)), W_norm, label='W norm')
# plt.axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')

# Add labels and legend
plt.xlabel('Iteration')
plt.ylabel('Norm value of W')
plt.title('W norm with Fixed Length MLP on verysmalldataset S='+ str(s_len))
plt.legend()

# Show the plot
plt.savefig('./figures/FixedLengthMLPs_Wnorm_'+ str(s_len)+'.png')
plt.show()

# exit(0)

# SVM

uniq_hs = tsk.get_unique()
uniq_embeds_dict = model.get_embeddings(uniq_hs, tsk.v_ctx2v)
# print(embeds)
support_set_pr = tsk.support_set_pr
support_set_sampled = tsk.support_set_sampled
ctx_dict = tsk.ctx_dict
if init_from == "mlp":
    svm = MultiLabelSVM(
    emb_dict=uniq_embeds_dict, support_dict=support_set_sampled, prob_dict=support_set_pr, d=model.d_output, v_nt=tsk.v_nt)
elif init_from == "tfm":
    svm = MultiLabelSVM(
        emb_dict=uniq_embeds_dict, support_dict=support_set_sampled, prob_dict=support_set_pr, d=d_decode,
        v_nt=tsk.v_nt)

w_fin = svm.find_Wfin()
print(w_fin)
w_star = svm.find_WStar()
print(w_star)

# set W = W_fin+infW* and compute entropy

svm_losses, plot_x  = [], []
embeds = model.forward_embedding(X)
for i in np.logspace(0.1, 3, 20, endpoint=True):
    W = torch.Tensor(w_fin + i * w_star)
    logits = embeds@W.T
    ce_loss = F.cross_entropy(logits, Y)
    print(f"alpha={i} | loss {ce_loss:.4f} | entropy {tsk_entropy:.2f} ")
    svm_losses.append(ce_loss.detach().numpy())
    plot_x.append(i)
#

# embedding structure
plt.clf()
plt.xscale('log')
plt.plot(plot_x, svm_losses, label='CE loss')
plt.axhline(y=tsk_entropy, color='r', linestyle='--', label='Dataset entropy')

# Add labels and legend

plt.xlabel('alpha')
plt.ylabel('loss')
plt.title('Cross Entropy loss with Fixed Length embedding and SVM verysmalldataset S='+ str(s_len))
plt.legend()

# Show the plot
plt.savefig('./figures/FixedLengthEmbedsSVMs_'+ str(s_len)+'.png')
plt.show()

# embedding geometry
sim_ctx = [[b'\x03\x00\x0e\x00\x04\x00&\x00\x06\x00' , b'\x03\x00\x0e\x00\x04\x00!\x00\x06\x00'], [b'\x03\x00\x18\x00\x04\x00\n\x00\x05\x00' ,  b'\x03\x00\x08\x00\x04\x00\x1b\x00\x05\x00'], [b'\x03\x00%\x00\x04\x00\x1a\x00\x06\x00' , b'\x03\x00\x08\x00\x04\x00\x13\x00\x06\x00']]
for k1, k2 in sim_ctx:
    sim_dist = np.linalg.norm(uniq_embeds_dict[k1] - uniq_embeds_dict[k2])
    diff_dists = []
    for k3, h3 in uniq_embeds_dict.items():
        if k3 == k1 or k3 == k2:
            continue
        diff_dists.append(np.linalg.norm(uniq_embeds_dict[k1] - uniq_embeds_dict[k3]))
        diff_dists.append(np.linalg.norm(uniq_embeds_dict[k2] - uniq_embeds_dict[k3]))
    diff_dist = np.array(diff_dists).mean()
    print(f"{k1+k2} | sim_dist {sim_dist:.4f} | diff_dist {diff_dist:.2f} | rate = {sim_dist/diff_dist:.2f}")




