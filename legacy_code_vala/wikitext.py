"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

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

import matplotlib.pylab as plt
from datasets import load_dataset


#DATA_CACHE_DIR = "/Users/yizezhao/Documents/Datasets"
DATA_CACHE_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/WikiText"
DATA_PROCESS_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/WikiText/WikiText_processing_files"



def train_vocab(vocab_size, target_tokens, num_stories = 2000):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_PROCESS_DIR, f"tok{vocab_size}")

    dataset = load_dataset("wikitext", cache_dir = "/arc/project/st-cthrampo-1/vala/WikiText")
    dataset = dataset["train"]

    wiki_file = os.path.join(DATA_CACHE_DIR, "wiki.txt")
    with open(wiki_file, "w", encoding="utf-8") as of:
        
        for text in tqdm(dataset[0:num_stories]['text']):
            text = text.strip()
            of.write(text + "\n")
    print(f"Size is: {os.path.getsize(wiki_file) / 1024 / 1024:.2f} MB")

    print("Will now train the vocab...")

    # print(wiki_file)
    # input()
    spm.SentencePieceTrainer.train(input=wiki_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size, 
                                   user_defined_symbols=target_tokens)

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_PROCESS_DIR, f"tok{vocab_size}.model")


def tokenize_one(vocab_size, num_stories, Train = True):

    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)

    dataset = load_dataset("wikitext", cache_dir = "/arc/project/st-cthrampo-1/vala/WikiText")
    dataset = dataset["train"]

    all_tokens = []

    if Train:
        for text in tqdm(dataset[0:num_stories]['text']):
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
    else:
        for text in tqdm(dataset[-num_stories:]['text']):
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
    
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename

    if Train:
        bin_basename = "Wikitext_voc" + str(vocab_size) + "_stor" + str(num_stories)+ ".bin" 
    else:
        bin_basename = "Wikitext_voc" + str(vocab_size) + "_stor" + str(num_stories)+ "_Test.bin" 
    
    tokenized_filename = os.path.join(DATA_PROCESS_DIR, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")



def pretokenize_target_stories(vocab_size, num_stories, num_stories_Test = 100):

    tokenize_one(vocab_size, num_stories, Train = True)
    print("Done.")




class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source, num_stories, AR_training, Train = "True"):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        self.iteration = 0
        self.num_batches = 0
        self.AR_training = AR_training
        self.Train = Train

        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        self.rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        if self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            # bin_dir = os.path.join(DATA_PROCESS_DIR, f"tok{self.vocab_size}")
            bin_dir = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/WikiText/WikiText_processing_files/"
            if Train:
                shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*_voc" + str(vocab_size) + "_stor" + str(num_stories) + ".bin")))
            else:
                shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*_voc" + str(vocab_size) + "_stor" + str(num_stories) + "_Test" + ".bin")))
            # shard_filenames = [bin_dir + "data00_voc" + str(vocab_size) + "_stor" + str(num_stories) + ".bin"]
        # train/test split. let's use only shard 0 for test split, rest train
        # shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        
        self.rng.shuffle(shard_filenames)
        for shard in shard_filenames:
            # open the dataset for reading but keep it on disk with memmap
            self.m = np.memmap(shard, dtype=np.uint16, mode="r")

            self.indices_to_sample_from = [i for i in range(0, len(self.m) - self.max_seq_len)]
       
            self.num_batches = len(self.indices_to_sample_from) // self.max_seq_len

            assert self.num_batches > 0, "this shard is way too small? investigate."
        
            

    def __iter__(self):
        print("Total number of training samples: " + str(len(self.indices_to_sample_from)))

        self.rng.shuffle(self.indices_to_sample_from)
        for ix in self.indices_to_sample_from:
            start = ix
            end = start + self.max_seq_len + 1
            chunk = torch.from_numpy((self.m[start:end]).astype(np.int64))

            if self.AR_training:
                x = chunk[:-1]
                y = chunk[1:]
            else:
                x = chunk[:-1]
                y = chunk[-1]
            yield x, y

    def __next__(self):
        if self.iteration < self.num_batches:
            sample = self.__iter__()
            self.iteration += 1
            return sample
        else:
            # Raise StopIteration when the iteration is complete
            self.rng.shuffle(self.indices_to_sample_from)
            self.iteration = 0
            raise StopIteration
            
# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=128, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--num_stories", type=int, default=10000, help="Number of stories to use")
    parser.add_argument("--context_length", type=int, default=16, help="Number of stories to use")
    args = parser.parse_args()


    if args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size, target_tokens = [], num_stories = args.num_stories)
        pretokenize_target_stories(vocab_size=args.vocab_size, num_stories = args.num_stories, num_stories_Test = 100)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
