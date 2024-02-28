'''
1. Train tokenizer(find(define) all unique tokens and sort(?))
2. Tokenize
3. dataloader
4. other text handlers
'''

import glob
import os
import random
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import Counter
import string
from tokenizer import Tokenizer
from separability import *


TOKENIZER_DIR = "./tokenizer"
DATA_DIR = "./data"

# class sequence_sampler():
#     def __init__(self, data_file, m=10, n=100, s_max=5, T=6, v_ctx=50, v_ntp=50):
#         self.tok_file = data_file
#         self.m = m
#         self.n = n
#         self.s_max = s_max
#         self.ctx_length = T-1
#         self.pr = np.zeros((m, v_ntp))
#
#     def read_in():
#
#         return
#     return

def get_tokenizer_model_path(vocab_size):
    if vocab_size == 0:
        return None
    else:
        return os.path.join(TOKENIZER_DIR, f"tok{vocab_size}.model")

def tokenize_one(vocab_size, pretok_file):

    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(pretok_file, "r") as f:
        pretok_text = f.readlines()
    all_tokens = []
    for text in tqdm(pretok_text):
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = enc.encode(text, bos=True, eos=True)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    # calculate the output filename
    # save .bin files into a new tok{N} directory
    binfile_name = pretok_file.replace('.txt', '.bin')
    #tokenized_filename = os.path.join(DATA_DIR, binfile_name)
    # write the bytes
    with open(binfile_name, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {binfile_name}, average seqlen: {avg_seq_len:.2f}")

def cnt_unique_words(data_file, save_pretokenize=True, position=None):
    save_file = None
    with open(data_file, "r") as f:
        text = f.read().lower().translate(str.maketrans('', '', string.punctuation))
        #text = text.replace('\n', ' .\n')
        unique_cnt = len(set(text.split()))

    if save_pretokenize:
        save_file = data_file.replace('.txt', '_pretok.txt')
        with open(save_file, "w") as fw:
            fw.write(text)
        print("pretok file saved at: ", save_file)
    return unique_cnt, save_file

def cnt_unique_words_tok(tok_file, vocab_size,  positions):
    tokenizer_file = get_tokenizer_model_path(vocab_size=vocab_size)

    positions = np.array(positions, dtype=np.uint16)

    m = np.memmap(tok_file, dtype=np.uint16, mode="r")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
    split_id = sp.PieceToId("<s>")

    split_ids = np.where(m==split_id)
    split_ids = np.asarray(split_ids, dtype=np.uint16)
    position_toks = []
    for pos in positions:
        position_toks.extend(m[split_ids+pos])
    unique_toks = (np.unique(position_toks))
    unique_cnt = len(unique_toks)

    return unique_cnt, unique_toks

def train_vocab(data_file, vocab_size=2000, set_vocab_size=True, model_type = 'word'):
    """
    Trains a custom sentencepiece tokenizer on a customized dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    if set_vocab_size == False:
        # count unique words
        vocab_size, pretok_file = cnt_unique_words(data_file)
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(TOKENIZER_DIR, f"tok{vocab_size}")
    # spm.SentencePieceTrainer.train(input=tiny_file,
    #                                model_prefix=prefix,
    #                                model_type="bpe",
    #                                vocab_size=vocab_size,
    #                                self_test_sample_size=0,
    #                                input_format="text",
    #                                character_coverage=1.0,
    #                                num_threads=os.cpu_count(),
    #                                split_digits=True,
    #                                allow_whitespace_only_pieces=True,
    #                                byte_fallback=True,
    #                                unk_surface=r" \342\201\207 ",
    #                                normalization_rule_name="identity")

    spm.SentencePieceTrainer.train(input=pretok_file,
                                   model_prefix=prefix,
                                   model_type=model_type,
                                   vocab_size=vocab_size )


    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


class Task:
    def __init__(self, batch_size, device, **dataset_kwargs):
        self.ds = FixedTDataset(**dataset_kwargs)
        self.emp_entropy = self.ds.data_entropy
        self.device = device
        self.batch = batch_size

        self.v_ctx = self.ds.v_ctx
        self.v_nt = self.ds.v_nt

        self.ctx_dict = self.ds.ctx_dict
        self.support_set_repeats = self.ds.support_set_repeats
        self.support_set_pr = self.ds.support_set_pr
        self.support_set_sampled = self.ds.support_set_sampled
        self.t = dataset_kwargs['T'] - 1

        self.v_ctx2v = self.ds.v_ctx2v

    def iter_batches(self):
        dl = torch.utils.data.DataLoader(self.ds, pin_memory=True)
        for x, y in dl:
            x = x[0].to(self.device, non_blocking=True)
            y = y[0].to(self.device, non_blocking=True)
            yield x, y

    def get_unique(self):
        uniq_hs_in_v_ctx = torch.from_numpy(self.ds.get_unique_ctxs()).to(self.device, non_blocking=True)
        return uniq_hs_in_v_ctx


class FixedTDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, T, tok_file, s_len,  vocab_size, bos, eos, if_batch=False):
        super().__init__()
        #self.split = split
        self.tok_file = tok_file
        # self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.T = T
        self.bos = bos
        self.eos = eos
        self.s = s_len
        self.if_batch = if_batch

        self.v2v_ctx = -1 * np.ones(vocab_size)
        self.v2v_nt = -1 * np.ones(vocab_size)
        self.v_ctx2v = -1 * np.ones(vocab_size)
        self.v_nt2v = -1 * np.ones(vocab_size)
        self.v = -1
        self.v_ctx = -1
        self.v_nt = -1

        self.ctx_dict = None
        self.support_set_repeats = None
        self.support_set_pr = None
        self.support_set_sampled = None

        self.sampled_data_file = os.path.join(DATA_DIR, "sampled.npy")
        self.data_entropy = self.sampler(equal_s = True, repeat_range = 5, save_pretok=self.sampled_data_file)

    def __iter__(self):
        # get worker info within a DataLoader
        # open the dataset for reading but keep it on disk with memmap
        seqs = np.load(self.sampled_data_file).astype(np.int64)

        while 1:
            if self.if_batch:
                for ix in range(seqs.shape[0]):
                    chunk = torch.from_numpy((seqs[ix]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[-1]
                    yield x, y
            else:
                x_in_v = seqs[:, :-1]
                x_in_v_ctx = self.v2v_ctx[x_in_v].astype(np.int64)
                y_in_v = seqs[:,-1]
                y_in_v_ctx = self.v2v_nt[y_in_v].astype(np.int64)
                yield x_in_v_ctx, y_in_v_ctx

    def get_unique_ctxs(self):
        m = len(self.ctx_dict.keys())
        t = self.T-1
        uniq_hs = np.zeros(shape=(m, t))
        i = 0
        for k, v in self.ctx_dict.items():
            uniq_hs[i] = v
            i += 1
        uniq_hs = uniq_hs.astype(np.int64)
        uniq_hs_in_v_ctx = self.v2v_ctx[uniq_hs].astype(np.int64)
        #return uniq_hs_in_v_ctx, self.v2v_ctx
        return uniq_hs_in_v_ctx

    def sampler(self, equal_s = True, repeat_range = 5, save_pretok="./sampled.npy"):
        # very small dataset 1 frequent labels
        cheatlist = np.arange(start=41, stop=51)
        # cheatlist = []
        # 1 get unique sequences
        ctx_len = self.T - 1
        toks = np.memmap(self.tok_file, dtype=np.uint16, mode="r")
        split_ids = np.where(toks == self.bos)
        ctx_seqs, support_set_dict = dict(), dict()
        #ctx_counter = 0
        for i, bos in enumerate(split_ids[0]):
            ctx = toks[bos+1:bos+self.T]
            nt = toks[bos+self.T]
            # convert ctx to hashable
            ctx_bytes = ctx.tobytes()
            # read in unique
            if ctx_bytes not in ctx_seqs.keys():
                ctx_seqs[ctx_bytes] = ctx
                support_set_dict[ctx_bytes] = []
                #ctx_counter += 1
            support_set_dict[ctx_bytes].append(nt)


        # 2 sample support set(make sure frequent nts are in the set)
        m = len(support_set_dict.keys())
        support_set_sampled = dict()
        support_set_repeats = dict()
        support_set_pr_dict = dict()
        conditional_entropy = 0
        n = 0
        for k, full_support in support_set_dict.items():
            if equal_s:
                s = self.s
            else:
                s = np.random.randint(1, self.s+1)
            # check if it contains the word we want to choose, prioritize these words when sampling
            common = np.sort(np.intersect1d(cheatlist, full_support))
            common_len = len(common)
            if common_len == 0:
                support_set_sampled[k] = np.random.choice(full_support, s, replace=False)
            elif common_len >= s:
                support_set_sampled[k] = common[:s]
            else:
                setdiff = np.setdiff1d(full_support, cheatlist, assume_unique=True)
                support_set_sampled[k] = np.array(common.tolist() + np.random.choice(setdiff, s-common_len, replace=False).tolist())
            # sanity check
            assert len(support_set_sampled[k]) == s

            # 3 sample appearing times
            repeats = np.random.randint(1, high=repeat_range, size=s)
            support_set_repeats[k] = np.repeat(support_set_sampled[k], repeats)

            # 4 calculate and construct probability matrix
            support_set_pr_dict[k] = repeats/repeats.sum()

            # 4.b calculate and save empirical entropy
            conditional_entropy -= np.sum(support_set_pr_dict[k] * np.log(support_set_pr_dict[k])) * repeats.sum()

            n += sum(repeats)

        # 5 construct complete dataset / save a temp tokenized file(?)
        seqs = np.zeros(shape=(n, self.T))
        i = 0
        for k, ctx in ctx_seqs.items():
            for s in range(len(support_set_repeats[k])):
                seqs[i][0:self.T-1] = ctx
                seqs[i][self.T-1] = support_set_repeats[k][s]
                i += 1

        # find mapping array
        #TODO： (THIS IS SO STUPID SHOULD COME UP WITH A BETTER WAY TO DO THIS!!!)
        x_v = seqs[:, :-1]
        # this is ctx2v, i think
        x_v_unique = np.unique(x_v).astype(np.int64)
        y_v = seqs[:, -1]
        y_v_unique = np.unique(y_v).astype(np.int64)

        for _v, _map in enumerate(x_v_unique):
            self.v2v_ctx[_map] = _v
        for _v, _map in enumerate(y_v_unique):
            self.v2v_nt[_map] = _v

        self.v_ctx2v = x_v_unique
        self.v_ctx = len(x_v_unique)
        self.v_nt2v = y_v_unique
        self.v_nt = len(y_v_unique)

        # write the bytes
        with open(save_pretok, "wb") as f:
            #f.write(seqs.tobytes())
            np.save(f, seqs)

        # 7 calculate entropy of this dataset and save for comparing
        conditional_entropy = 1/(n) * conditional_entropy
        support_set_sampled_vnt = dict()
        for k, v in support_set_sampled.items():
            support_set_sampled_vnt[k] = self.v2v_nt[v].astype(np.int64)

        self.ctx_dict = ctx_seqs
        self.support_set_repeats = support_set_repeats
        self.support_set_pr = support_set_pr_dict
        self.support_set_sampled = support_set_sampled_vnt

        return conditional_entropy

    # def v_mapping(self):




class ARDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        #bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y
if __name__ == '__main__':
    data_file = "./data/verysmallset.txt"
    pretok_file = "./data/verysmallset_pretok.txt"
    vocab_size, pretok_file = cnt_unique_words(data_file)
    print(vocab_size)
    #vocab_size = 153


    # train tokenizer
    train_vocab(data_file, set_vocab_size=False, model_type='word')

    # tokenize one file
    tokenize_one(vocab_size=vocab_size, pretok_file=pretok_file)

    # cnt/analyze tokenized file
    unq_cnt, unq_toks_pred = cnt_unique_words_tok("./data/verysmallset_pretok.bin", vocab_size=vocab_size, positions=[6])
    print("next token vocabulary at 6th token: ", unq_cnt)

    unq_cnt, unq_toks_ctx = cnt_unique_words_tok("./data/verysmallset_pretok.bin", vocab_size=vocab_size, positions=range(5))
    print("next token vocabulary at 0-5th token: ", unq_cnt)

    #sanity checks:
    tokenizer_file = get_tokenizer_model_path(vocab_size=vocab_size)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_file)

    pred_toks = [sp.IdToPiece(id) for id in unq_toks_pred.tolist()]
    ctx_toks = [sp.IdToPiece(id) for id in unq_toks_ctx.tolist()]

    print("pred_toks: ", pred_toks)
    print("ctx_toks: ", ctx_toks)

