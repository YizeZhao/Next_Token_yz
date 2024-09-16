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
import re
from tokenizer import Tokenizer
from separability import *


TOKENIZER_DIR = "./tokenizer"
DATA_DIR = "./data"

random.seed(1634)
np.random.seed(1634)


def get_rank_by_svd(A):
    AAT = np.dot(A, A.T)
    U_aat, singular_values_aat, Vt_aat = np.linalg.svd(AAT)
    rank_aat = np.sum(singular_values_aat > 1e-10)
    return rank_aat
def get_tokenizer_model_path(vocab_size):
    if vocab_size == 0:
        return None
    else:
        return os.path.join(TOKENIZER_DIR, f"tok{vocab_size}.model")

def tokenize_one(tokenizer_model_path, pretok_file, tok_type):

    # tokenizer_model = get_tokenizer_model_path(vocab_size)
    tokenizer_model = tokenizer_model_path
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
    if tok_type == "char" or tok_type == "word":
        binfile_name = pretok_file.replace('.txt', f'_{tok_type}''.bin')
    else:
        binfile_name = pretok_file.replace('.txt', f'_{vocab_size}_{tok_type}''.bin')
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
        # text = f.read().lower().translate(str.maketrans('', '', string.punctuation))
        text = f.read().lower()
        text = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)

        text = re.sub(r'^$\n', '', text, flags=re.MULTILINE)
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
    data_file_id = data_file.split("/")[-1].split(".")[0]


    # output file prefix path for sentencepiece
    if model_type == 'word' or model_type == 'char':
        prefix = os.path.join(TOKENIZER_DIR, f"tok_{data_file_id}_{model_type}")
    else:
        prefix = os.path.join(TOKENIZER_DIR, f"tok_{data_file_id}_{vocab_size}")
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
                                   vocab_size=vocab_size,
                                   use_all_vocab=True)
                                   #  vocab_size = 703 )


    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")
    return f"{prefix}.model"



class Task:
    def __init__(self, batch_size, device, x_type, **dataset_kwargs):
        self.ds = FixedTDataset(**dataset_kwargs)
        self.emp_entropy = self.ds.data_entropy
        self.device = device
        self.batch = batch_size

        self.v_ctx = self.ds.v_ctx
        self.v_nt = self.ds.v_nt

        #self.if_ufm = ufm
        self.m = self.ds.m
        self.n = self.ds.n
        self.x_type = x_type

        self.ctx_dict = self.ds.ctx_dict
        self.support_set_repeats = self.ds.support_set_repeats
        self.support_set_pr = self.ds.support_set_pr
        self.support_set_sampled = self.ds.support_set_sampled
        self.t = dataset_kwargs['T'] - 1
        self.if_batch = dataset_kwargs["if_batch"]

        self.v_ctx2v = self.ds.v_ctx2v
        self.v_nt2v = self.ds.v_nt2v

    def iter_batches(self):
        if self.if_batch == False:
            dl = torch.utils.data.DataLoader(self.ds, pin_memory=True, batch_size=int(self.n), shuffle=False)
        #     for x, y in dl:
        #         # x = x[0].to(self.device, non_blocking=True)
        #         x = x[0].to(self.device, non_blocking=True).type(self.x_type)
        #         y = y[0].to(self.device, non_blocking=True).type(torch.LongTensor)
        #         yield x, y
        else:
            dl = torch.utils.data.DataLoader(self.ds, pin_memory=True, batch_size=self.batch, shuffle=False)

        #     for i, (x, y) in enumerate(dl):
        #         x = x.to(self.device, non_blocking=True).type(self.x_type)
        #         y = y.to(self.device, non_blocking=True).type(torch.LongTensor)
        #         yield x, y

        return dl



    def get_unique(self):
        uniq_hs_in_v_ctx = torch.from_numpy(self.ds.get_unique_ctxs()).to(self.device, non_blocking=True)
        return uniq_hs_in_v_ctx

class FixedTDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, T, tok_file, s_len,  vocab_size, bos, eos, if_batch=False, if_ufm=False, predefined=False, balanced=False, set_s="equal", repeat_range=5, save_pretok=None):
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
        self.n = 0
        self.m = 0
        self.if_ufm = if_ufm
        self.predefined = predefined
        self.toy_balanced = balanced

        self.v2v_ctx = -1 * np.ones(vocab_size)
        self.v2v_nt = -1 * np.ones(vocab_size)
        self.v_ctx2v = -1 * np.ones(vocab_size)
        self.v_nt2v = -1 * np.ones(vocab_size)
        self.v = -1
        self.v_ctx = -1
        self.v_nt = -1

        self.iteration = 0

        self.ctx_dict = None
        self.support_set_repeats = None
        self.support_set_pr = None
        self.support_set_sampled = None
        self.ctx_id_dict = None

        self.sampled_data_file = save_pretok
        self.data_entropy = self.sampler(set_s = set_s, repeat_range = repeat_range, save_pretok=self.sampled_data_file)
    def __iter__(self):
        # get worker info within a DataLoader
        # open the dataset for reading but keep it on disk with memmap
        seqs = np.load(self.sampled_data_file).astype(np.int64)

        # while 1:
        #     for i in range(0, len(seqs), self.T):
        i = 0
        if self.if_batch:
            for ix in range(seqs.shape[0]):
                i += 1
                chunk = ((seqs[ix]).astype(np.int64))
                if self.if_ufm:
                    x = chunk[:-1].astype(np.float64)
                    y = chunk[-1].astype(np.int64)
                    y_in_v_ctx = self.v2v_nt[y].astype(np.int64)
                    yield x, y_in_v_ctx

                else:
                    x_in_v = chunk[:-1]
                    x_in_v_ctx = self.v2v_ctx[x_in_v].astype(np.int64)
                    y_in_v = np.array(chunk[-1])
                    y_in_v_ctx = self.v2v_nt[y_in_v].astype(np.int64)
                    # print(i)
                    # print(x_in_v_ctx.shape)
                    yield x_in_v_ctx, y_in_v_ctx
            # for ix in range(seqs.shape[0]):
            #     chunk = torch.from_numpy((seqs[ix]).astype(np.int64))
            #     x = chunk[:-1]
            #     y = chunk[-1]
            #     yield x, y
        else:
            if self.if_ufm:
                x = seqs[:, :-1].astype(np.float64)
                y = seqs[:, -1]
                y_in_v_ctx = self.v2v_nt[y].astype(np.int64)
                yield x, y_in_v_ctx

            else:
                x_in_v = seqs[:, :-1]
                x_in_v_ctx = self.v2v_ctx[x_in_v].astype(np.int64)
                y_in_v = seqs[:,-1]
                y_in_v_ctx = self.v2v_nt[y_in_v].astype(np.int64)
                yield x_in_v_ctx, y_in_v_ctx

    def __next__(self):
        if self.iteration >= self.n:
            raise StopIteration
        self.iteration += 1
        return self.__iter__()
    def get_ufm_data(self):
        seqs = np.zeros(shape=(self.n, self.m+1))
        i = 0
        for k, id_ctx_dict in self.ctx_dict.items():
            for s in range(len(self.support_set_repeats[k])):
                seqs[i][id_ctx_dict["id"]] = 1
                seqs[i][self.m] = self.support_set_repeats[k][s]
                i += 1
        return seqs

    def get_unique_ctxs(self):
        if self.if_ufm:
            return np.eye(self.m)

        m = len(self.ctx_dict.keys())
        t = self.T-1
        uniq_hs = np.zeros(shape=(m, t))
        i = 0
        for k, v in self.ctx_dict.items():
            uniq_hs[v["id"]] = v["value"]
            i += 1
        uniq_hs = uniq_hs.astype(np.int64)
        uniq_hs_in_v_ctx = self.v2v_ctx[uniq_hs].astype(np.int64)
        #return uniq_hs_in_v_ctx, self.v2v_ctx
        return uniq_hs_in_v_ctx

    def sample_predefined(self, ):
        self.m = 3
        support_set_sampled = {'a':[0,1], 'b':[0,2], 'c': [0,2]}
        support_set_repeats = {'a':[1,1], 'b':[1,1], 'c': [1,1]}
        support_set_pr_dict = {'a':[0.5,0.5], 'b':[0.5,0.5], 'c': [0.5,0.5]}

    def sampler(self, set_s = "equal", repeat_range = 5, save_pretok="./sampled.npy"):

        # very small dataset 1 frequent labels
        cheatlist = np.arange(start=41, stop=51)
        # cheatlist = []
        # 1 get unique sequences
        ctx_len = self.T - 1
        toks = np.memmap(self.tok_file, dtype=np.uint16, mode="r")
        split_ids = np.where(toks == self.bos)
        ctx_seqs, support_set_dict = dict(), dict()
        ctx_counter = 0
        for i, bos in enumerate(split_ids[0]):
            ctx = toks[bos+1:bos+self.T]
            nt = toks[bos+self.T]
            # convert ctx to hashable
            ctx_bytes = ctx.tobytes()
            # read in unique
            if ctx_bytes not in ctx_seqs.keys():
                ctx_seqs[ctx_bytes] = {}
                ctx_seqs[ctx_bytes]["value"] = ctx
                ctx_seqs[ctx_bytes]["id"] = ctx_counter
                support_set_dict[ctx_bytes] = []
                ctx_counter += 1
            support_set_dict[ctx_bytes].append(nt)


        # 2 sample support set(make sure frequent nts are in the set)
        self.m = len(support_set_dict.keys())
        support_set_sampled = dict()
        support_set_repeats = dict()
        support_set_pr_dict = dict()
        conditional_entropy = 0
        n = 0
        tempc = 0
        for k, full_support_raw in support_set_dict.items():
            tempc += 1
            full_support = list(set(full_support_raw))
            full_support_len = len(full_support)
            if set_s == "equal":
                s = self.s
                if full_support_len < s:
                    full_support = full_support + list(range(3, s-full_support_len+3))
            elif set_s == "random":
                s = np.random.randint(1, full_support_len+1)

            elif set_s == "original":
                s = full_support_len
            # check if it contains the word we want to choose, prioritize these words when sampling

            common = np.sort(np.intersect1d(cheatlist, full_support))
            common_len = len(common)

            if self.toy_balanced:
                s = self.s
                #support_set_sampled[k] = np.random.choice(np.arange(0, 100), s, replace=False)
                support_set_sampled[k] = np.arange(s) + tempc//10 * s
            elif common_len == 0:
                support_set_sampled[k] = np.random.choice(full_support, s, replace=False)
            elif common_len >= s:
                support_set_sampled[k] = common[:s]
            else:
                setdiff = np.setdiff1d(full_support, cheatlist, assume_unique=True)
                support_set_sampled[k] = np.array(common.tolist() + np.random.choice(setdiff, s-common_len, replace=False).tolist())
            # sanity check
            assert len(support_set_sampled[k]) == s

            if len(support_set_sampled[k]) != len(np.unique(support_set_sampled[k])):
                print("duplicate in support set")

            # 3 sample appearing times
            repeats = np.random.randint(1, high=repeat_range, size=s)
            # repeats = np.full(s, 3)
            support_set_repeats[k] = np.repeat(support_set_sampled[k], repeats)

            # 4 calculate and construct probability matrix
            support_set_pr_dict[k] = repeats/repeats.sum()

            # 4.b calculate and save empirical entropy
            conditional_entropy -= np.sum(support_set_pr_dict[k] * np.log(support_set_pr_dict[k])) * repeats.sum()

            n += sum(repeats)

        self.n = n
        # 5 construct complete dataset / save a temp tokenized file(?)

        seqs = np.zeros(shape=(n, self.T))
        i = 0
        for k, id_ctx_dict in ctx_seqs.items():
            for s in range(len(support_set_repeats[k])):
                seqs[i][0:self.T-1] = id_ctx_dict["value"]
                seqs[i][self.T-1] = support_set_repeats[k][s]
                i += 1

        # count the number of unique values in support_set_sampled


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

        # 7 calculate entropy of this dataset and save for comparing
        conditional_entropy = 1/(n) * conditional_entropy
        support_set_sampled_vnt = dict()
        for k, v in support_set_sampled.items():
            support_set_sampled_vnt[k] = self.v2v_nt[v].astype(np.int64)

        self.ctx_dict = ctx_seqs
        self.support_set_repeats = support_set_repeats
        self.support_set_pr = support_set_pr_dict
        self.support_set_sampled = support_set_sampled_vnt

        if self.if_ufm:
            seqs = self.get_ufm_data()

        # write the bytes
        with open(save_pretok, "wb") as f:
            #f.write(seqs.tobytes())
            np.save(f, seqs)

        return conditional_entropy

    # def v_mapping(self):



class ARDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, max_seq_len, vocab_size, pretok_filename):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # self.pretok_path = os.path.join(DATA_DIR, pretok_filename)
        self.pretok_path = pretok_filename

        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42
        self.rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")

        self.m = np.memmap(self.pretok_path, dtype=np.uint16, mode="r")

        # self.num_batches = len(self.m) // self.max_seq_len
        # self.num_batches -= 1  # drop the last partial batch
        # assert self.num_batches > 0, "this shard is way too small? investigate."

        self.num_batches = len(self.m) - self.max_seq_len - 1
        self.indices_to_sample_from = list(range(self.num_batches))

    def __iter__(self):
        # print("Total number of training samples: " + str(len(self.indices_to_sample_from)))

        self.rng.shuffle(self.indices_to_sample_from)
        for ix in self.indices_to_sample_from:
            # start = ix * self.max_seq_len
            start = ix
            end = start + self.max_seq_len + 1
            # calling .astype will copy the data into a new numpy array, now in RAM
            chunk = torch.from_numpy((self.m[start:end]).astype(np.int64))
            x = chunk[:-1]
            y = chunk[1:]
            yield x, y

    def __next__(self):
        if self.iteration >= self.num_batches:
            raise StopIteration
        self.iteration += 1
        return self.__iter__()



if __name__ == '__main__':
    data_file = "./data/tiny_extract_m404_3.txt"

    # data_file = "./data/tiny100lines.txt"
    vocab_size, pretok_file = cnt_unique_words(data_file)
    print(vocab_size)

    #vocab_size = 153


    # train tokenizer
    tok_type = 'word'
    tokenizer_model_path = train_vocab(data_file, set_vocab_size=False, model_type=tok_type)

    # tokenize one file
    tokenize_one(tokenizer_model_path=tokenizer_model_path, pretok_file=pretok_file, tok_type=tok_type)

    # # cnt/analyze tokenized file
    # unq_cnt, unq_toks_pred = cnt_unique_words_tok("./data/verysmallset_pretok.bin", vocab_size=vocab_size, positions=[6])
    # print("next token vocabulary at 6th token: ", unq_cnt)
    #
    # unq_cnt, unq_toks_ctx = cnt_unique_words_tok("./data/verysmallset_pretok.bin", vocab_size=vocab_size, positions=range(5))
    # print("next token vocabulary at 0-5th token: ", unq_cnt)

    # #sanity checks:
    # tokenizer_file = get_tokenizer_model_path(vocab_size=vocab_size)
    # print(tokenizer_file)
    # sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
    #
    # print(sp.IdToPiece(100))
    #
    # toks = [sp.IdToPiece(id) for id in range(vocab_size-1)]
    #
    # print(toks[:100])

    # pred_toks = [sp.IdToPiece(id) for id in unq_toks_pred.tolist()]
    # ctx_toks = [sp.IdToPiece(id) for id in unq_toks_ctx.tolist()]
    #
    # print("pred_toks: ", pred_toks)
    # print("ctx_toks: ", ctx_toks)

