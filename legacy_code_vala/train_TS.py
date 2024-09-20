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
import time as time 
import psutil

import pickle 
import gc
import tracemalloc
import resource
import sys
from pympler import asizeof


import mmap
import hashlib


DATA_CACHE_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/TinyStories"
DATA_PROCESS_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/TS_Scaling_processing_files"

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")


def train_vocab(vocab_size, target_tokens, num_stories = 2000):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_PROCESS_DIR, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 1

    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    dict_story_words = {}
    triple_words_dict = {}

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            
            for example in tqdm(data[0:num_stories]):
                
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")



    # 2) train the sentencepiece model
    print("Will now train the vocab...")

    spm.SentencePieceTrainer.train(input=tiny_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size, 
                                   user_defined_symbols=target_tokens)

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def tokenize_one(vocab_size, json_file, num_stories_start, num_stories_end, Train = True):

    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)

    with open(json_file, "r") as f:
        data = json.load(f)
    all_tokens = []

    if Train:
        for example in tqdm(data[num_stories_start:num_stories_end]):
            text = example["story"]
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
            all_tokens.append(vocab_size + 2)
    else:
        for example in tqdm(data[-num_stories:]):
            text = example["story"]
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
            all_tokens.append(vocab_size + 2)
    
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = json_file.replace(".json", ".bin")
    else:
        shard_basename = os.path.basename(json_file)
        if Train:
            bin_basename = shard_basename.replace(".json", "_voc" + str(vocab_size) + "_stor" + str(num_stories)+ ".bin")
        else:
            bin_basename = shard_basename.replace(".json", "_voc" + str(vocab_size) + "_stor" + str(num_stories) + "_Test" + ".bin")
        tokenized_filename = os.path.join(DATA_PROCESS_DIR, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

    return all_tokens


def pretokenize_target_stories(vocab_size, num_stories_start, num_stories_end):
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    tokenize_one(vocab_size, DATA_CACHE_DIR + "/TinyStories_all_data/data00.json", num_stories_start, num_stories_end, Train = True)
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
        if self.vocab_source == "llama2":
            # the .bin files are right along the .json files
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/TS_Scaling_processing_files/"
            if Train:
                shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*_voc" + str(vocab_size) + "_stor" + str(num_stories) + ".bin")))
            else:
                shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*_voc" + str(vocab_size) + "_stor" + str(num_stories) + "_Test" + ".bin")))

        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        
        self.rng.shuffle(shard_filenames)
        for shard in shard_filenames:
            # open the dataset for reading but keep it on disk with memmap
            self.m = np.memmap(shard, dtype=np.uint16, mode="r")
            # indices = np.where(m)[vocab_size + 2]
            indices = [i for i, x in enumerate(self.m == vocab_size + 2) if x]
            indices.insert(0, True)
            num_examples = sum(indices[i] - indices[i-1] - self.max_seq_len for i in range(1,len(indices)))

            self.indices_to_sample_from = []
            for i in range(1, len(indices)):
                self.indices_to_sample_from.extend(range(indices[i-1] + 1, indices[i] - self.max_seq_len))
            # index = 1
            # # for i in range(1, len(indices)):
            # while index < len(indices):
            #     self.indices_to_sample_from.extend(range(indices[index-1] + 1, indices[index] - self.max_seq_len))
            #     index += self.max_seq_len // 2
       
            self.num_batches = len(self.indices_to_sample_from) // self.max_seq_len
            # self.num_batches = 100
            # num_batches -= 1  # drop the last partial batch
            assert self.num_batches > 0, "this shard is way too small? investigate."
        
        # rng.shuffle(self.indices_to_sample_from)
            

    def __iter__(self):
        print("Total number of training samples: " + str(len(self.indices_to_sample_from)))

        self.rng.shuffle(self.indices_to_sample_from)
        for ix in self.indices_to_sample_from:
            start = ix
            end = start + self.max_seq_len + 1
            # calling .astype will copy the data into a new numpy array, now in RAM
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
# public interface functions

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

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        return dl



def calc_data_context_info(context_length, vocab_size, vocab_source, AR_training, context_to_index_dict, pos_context_count, sup_set_counts_dict, context_entropy_vector, context_count_vector):

    ds = PretokDataset("train", context_length, vocab_size, vocab_source, num_stories, AR_training = AR_training)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1000, pin_memory=True, num_workers=0
    )

    entropy = 0
    entropy_pos = {}
    for pos in range(1, context_length + 1):
        entropy_pos[pos] = 0

    if len(context_to_index_dict.keys()) == 0:
        index_counter = 0
    else:   
        index_counter = max(context_to_index_dict.values())
        # index_counter += 1 # !!!!!!!! CHECK IF THIS IS CORRECT AGAIN 
    

    # tracemalloc.start()

    # print("*" * 100)
    # print("At start")
    # print('RAM Used (GB):', psutil.virtual_memory()[3]/(1024**3))
    # print('MidPeak RAM Used (GB):', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2))
    # print("*" * 100)

    updated_context_indices = []

    start_time = time.time()
    sup_sets = set(sup_set_counts_dict.keys())
    with open('/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/Saved_S_P_matrices_2/Smatrix_Voc_' + str(args.vocab_size) + "_ctxLen_" + str(args.context_length) + '.bin', 'r+b') as f_S:
        mm_S = mmap.mmap(f_S.fileno(), 0, access=mmap.ACCESS_WRITE)
        # Interpret the memory as a NumPy array
        S_matrix = np.frombuffer(mm_S, dtype=np.bool_).reshape((100000000, vocab_size))
        
        with open('/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/Saved_S_P_matrices_2/Pmatrix_Voc_' + str(args.vocab_size) + "_ctxLen_" + str(args.context_length) + '.bin', 'r+b') as f_P:
            mm_P = mmap.mmap(f_P.fileno(), 0, access=mmap.ACCESS_WRITE)
            # Interpret the memory as a NumPy array
            P_matrix = np.frombuffer(mm_P, dtype=np.uint32).reshape((100000000, vocab_size))
            
            num_datapoints = 0
            print("Creating context dictionaries: ")
            for batch_idx, (X, Y) in tqdm(enumerate(dl)):
                for index in range(0, Y.shape[0]):
                    # Should this sum be here ? 
                    # num_datapoints += 1
                    x = X[index, :]
                    y = Y[index, :]
                
                    for i in range(1, context_length + 1):
                        context = x[:i]
                        token = y[i - 1]

                        pos_context_count[len(context)] += 1

                        context = "_".join([str(n) for n in context.tolist()])
                        # context = tuple(context.tolist())
                        token = token.item()

                        num_datapoints += int(1)

                        if context in context_to_index_dict.keys():
                            context_index = context_to_index_dict[context]
                            old_sup_set_tuple = tuple(S_matrix[context_index,:])

                            New_token_for_context = False
                            if S_matrix[context_index, token] == False:
                                New_token_for_context = True
                            
                            S_matrix[context_index, token] = True
                            P_matrix[context_index, token] += 1
                            
                            context_count_vector[context_index] += 1
                            updated_context_indices.append(context_index)

                            sup_set_tuple = tuple(S_matrix[context_index,:])
                            if sup_set_tuple not in sup_sets:
                                sup_set_counts_dict[sup_set_tuple] = 1
                                sup_set_counts_dict[old_sup_set_tuple] -= 1
                                sup_sets.add(sup_set_tuple)
                            else:
                                sup_set_counts_dict[sup_set_tuple] += 1
                                if New_token_for_context:
                                    sup_set_counts_dict[old_sup_set_tuple] -= 1
                            
                            if sup_set_counts_dict[old_sup_set_tuple] == 0:
                                del sup_set_counts_dict[old_sup_set_tuple]
                                sup_sets.remove(old_sup_set_tuple)

                        else:
                            context_to_index_dict[context] = int(index_counter)
                            # if S_matrix.shape[0] <= index_counter:
                            #     print("Updating size")
                            #     S_matrix = np.concatenate((S_matrix, np.zeros((100000,vocab_size), dtype=np.ushort)), axis=0)
                            
                            S_matrix[index_counter, token] = True
                            P_matrix[index_counter, token] += 1
                            
                            context_count_vector[index_counter] += 1
                            updated_context_indices.append(index_counter)
                            sup_set_tuple = tuple(S_matrix[index_counter,:])
                            index_counter += int(1)
                            
                            if sup_set_tuple not in sup_sets:
                                sup_set_counts_dict[sup_set_tuple] = 1
                                sup_sets.add(sup_set_tuple)
                            else:
                                sup_set_counts_dict[sup_set_tuple] += 1

                            
                        del context
                        del token
                    del x
                    del y
                    # gc.collect()

                del X
                del Y
                gc.collect()

            del dl
            del ds
            gc.collect()

    end_time = time.time() - start_time
    print("Parse Data took " + str(end_time) + " seconds!")


    start_time = time.time()
    for updated_context_index in tqdm(updated_context_indices):

        non_zero_tokens = tuple(S_matrix[updated_context_index,:].nonzero()[0])
        context_freq = context_count_vector[updated_context_index]
        
        context_entropy_vector[updated_context_index] = 0
        for token in non_zero_tokens:
            context_entropy_vector[updated_context_index] -= P_matrix[updated_context_index,token].item() / context_freq * np.log(P_matrix[updated_context_index,token].item() / context_freq)


    entropy = np.dot(context_entropy_vector, context_count_vector) / np.sum(context_count_vector)
    end_time = time.time() - start_time
    print("Entropy calculations took " + str(end_time) + " seconds!")


    # Stop tracing memory allocations
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # current, peak =  tracemalloc.get_traced_memory()
    print("Size of context_to_index_dict:" + str(asizeof.asizeof(context_to_index_dict) / (1024**2)))
    print("Size of sup_set_counts_dict:" + str(asizeof.asizeof(sup_set_counts_dict) / (1024**2)))
    
    del S_matrix
    del P_matrix
    mm_S.flush()
    mm_P.close()
    
  

    return_dict = {}

    # set_unique_sup_set = set()
    dict_unique_sup_set_counts = {}
    dict_unique_sup_set_unique_context_sets = {}
    dict_unique_sup_set_unique_context_sets_counts = {}
    num_unique_sup_sets = 0
    sup_set_size_dist = {}
    

    num_full_contexts = num_datapoints // context_length
    total_num_contexts = num_datapoints

    num_unique_contexts = index_counter - 1
    num_unique_sup_sets = len(sup_sets)
    sup_set_context_count_dist = np.sort(list(dict_unique_sup_set_counts.values()))
    sup_set_unique_context_count_dist = np.sort(list(dict_unique_sup_set_unique_context_sets_counts.values()))

    return_dict = {"entropy": entropy,
                   "entropy_pos": entropy_pos,
                   "total_num_contexts": total_num_contexts,
                   "num_full_contexts": num_full_contexts,
                   "num_unique_contexts": num_unique_contexts,
                   "num_unique_sup_sets": num_unique_sup_sets,
                   "sup_set_context_count_dist": sup_set_context_count_dist,
                   "sup_set_unique_context_count_dist": sup_set_unique_context_count_dist,
                   "sup_set_size_dist": sup_set_size_dist
                   }

    return sup_set_counts_dict, context_to_index_dict, context_entropy_vector, context_count_vector, pos_context_count, return_dict

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
    parser.add_argument("--vocab_size", type=int, default=1024, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=16, help="Number of stories to use")
    args = parser.parse_args()
    

    list_num_stories = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for i in range(2, 101):
        list_num_stories.append(i * 1000)
    train_vocab(vocab_size=args.vocab_size, target_tokens = [], num_stories = 10000)
    start_stories = 0
    
    # For S
    # Calculate the total size of the array in bytes
    # Create a file to map into memory (this will create a zero-filled file)
    total_bytes = 100000000 * args.vocab_size * np.dtype(np.bool_).itemsize
    with open('/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/Saved_S_P_matrices_2/Smatrix_Voc_' + str(args.vocab_size) + "_ctxLen_" + str(args.context_length) + '.bin', 'wb') as f_S:
        f_S.seek(total_bytes - 1)
        f_S.write(b'\0')

    total_bytes = 100000000 * args.vocab_size * np.dtype(np.uint32).itemsize
    with open('/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/Saved_S_P_matrices_2/Pmatrix_Voc_' + str(args.vocab_size) + "_ctxLen_" + str(args.context_length) + '.bin', 'wb') as f_P:
        f_P.seek(total_bytes - 1)
        f_P.write(b'\0')
    
    
    
    context_to_index_dict = {}

    pos_context_count = {}
    for pos in range(1, args.context_length + 1):
        pos_context_count[pos] = 0
    
    vocab_source = "custom"
    AR_training = True


    context_entropy_vector = np.zeros((100000000), dtype = np.double)
    context_count_vector = np.zeros((100000000), dtype = np.int32)

    all_stories_results_dict = {}
    sup_set_counts_dict = {}

    for num_stories in list_num_stories:
        print("Num Stories being processed: " + str(num_stories))
        pretokenize_target_stories(vocab_size=args.vocab_size, num_stories_start = start_stories, num_stories_end = num_stories)
        start_stories = num_stories

        # tracemalloc.start()
        start_time = time.time()
        sup_set_counts_dict, context_to_index_dict, context_entropy_vector, context_count_vector, pos_context_count, return_dict = calc_data_context_info(args.context_length, args.vocab_size, vocab_source, AR_training, context_to_index_dict, pos_context_count, sup_set_counts_dict, context_entropy_vector, context_count_vector)
        total_time = time.time() - start_time

        all_stories_results_dict[num_stories] = return_dict
        with open("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/logs_EntropyAndSup_2/Full_Results_TS_Voc_" + str(args.vocab_size) + "_ctxLen_" + str(args.context_length)   + '.pkl', 'wb') as f:
            pickle.dump(all_stories_results_dict, f)
        
        with open("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/logs_EntropyAndSup_2/sup_set_counts_dict_TS_Voc_" + str(args.vocab_size) + "_ctxLen_" + str(args.context_length)   + '.pkl', 'wb') as f:
            pickle.dump(sup_set_counts_dict, f)
        
        with open("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/logs_EntropyAndSup_2/context_to_index_dict_TS_Voc_" + str(args.vocab_size) + "_ctxLen_" + str(args.context_length)   + '.pkl', 'wb') as f:
            pickle.dump(context_to_index_dict, f)
        
        with open("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/logs_EntropyAndSup_2/context_entropy_vector_TS_Voc_" + str(args.vocab_size) + "_ctxLen_" + str(args.context_length)   + '.npy', 'wb') as f:
            np.save(f, context_entropy_vector)
        
        with open("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/logs_EntropyAndSup_2/context_count_vector_TS_Voc_" + str(args.vocab_size) + "_ctxLen_" + str(args.context_length)   + '.npy', 'wb') as f:
            np.save(f, context_count_vector)
        

        print("Entropy is " + str(return_dict["entropy"]))
        print("total_num_contexts: " + str(return_dict["total_num_contexts"]))
        print("num_unique_contexts: " + str(return_dict["num_unique_contexts"]))
        print("num_unique_sup_sets: " + str(return_dict["num_unique_sup_sets"]))
        # print("Time for operation: " + str(total_time / 60) + " minutes.")
        # print('RAM memory % used:', psutil.virtual_memory().percent)
        # print('RAM memory % free:', psutil.virtual_memory().free/(1024**3))
        # print('RAM Used (GB):', psutil.virtual_memory()[3]/(1024**3))
        process = psutil.Process(os.getpid())
        print('Physical RAM Used (GB):', process.memory_info().rss/(1024**3))
        print('Physical RAM % Used (GB):', process.memory_percent())
        print('MidPeak RAM Used (GB):', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2))
        print("_" * 100)

        

    