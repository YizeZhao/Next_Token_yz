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


#DATA_CACHE_DIR = "/Users/yizezhao/Documents/Datasets"
DATA_CACHE_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/TinyStories"
DATA_PROCESS_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/TinyStories/TinyStories_processing_files"

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


    # top_words = ["open", "clean", "young", "yellow"]
    dict_story_words = {}
    triple_words_dict = {}

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            
            # for example in data[0:num_stories]:
            for example in tqdm(data[0:num_stories]):
                
                # for word in example["instruction"]["words"]:
                #     if word in dict_story_words.keys():
                #         dict_story_words[word] += 1
                #     else:
                #         dict_story_words[word] = 1
                
                # words_set = "_".join(example["instruction"]["words"])
                # if word in triple_words_dict.keys():
                #     triple_words_dict[words_set] += 1
                # else:
                #     triple_words_dict[words_set] = 1
                
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")


    # dict_story_words = dict(sorted(dict_story_words.items(), key=lambda item: item[1]))
    # print(dict_story_words)
    # print(len(list(dict_story_words.keys())))

    # print(triple_words_dict)
    # print(len(triple_words_dict))
    # input()

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
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

    spm.SentencePieceTrainer.train(input=tiny_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size, 
                                   user_defined_symbols=target_tokens)

    # 3) optional cleanup, ask the user if they'd like to delete tiny.txt
    # dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    # if dec.lower() == "y":
    #     os.remove(tiny_file)
    #     print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def tokenize_one(vocab_size, json_file, num_stories, Train = True):

    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)


    with open(json_file, "r") as f:
        data = json.load(f)
    all_tokens = []

    if Train:
        for example in tqdm(data[0:num_stories]):
            text = example["story"]
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
            all_tokens.append(999)
    else:
        for example in tqdm(data[-num_stories:]):
            text = example["story"]
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
            all_tokens.append(999)
    
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = json_file.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        # bin_dir = os.path.join(DATA_PROCESS_DIR, f"tok{vocab_size}")
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


def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size):
    # how many shards we'll use to find target contexts
    num_shards = 1

    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    # if vocab_size > 0:
    #     # .bin files will be saved into tok{N} directory, create it once here
    #     bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    #     os.makedirs(bin_dir, exist_ok=True)
    #
    # # process all the shards in a process pool
    # fun = partial(process_shard, vocab_size=vocab_size)
    # with ProcessPoolExecutor() as executor:
    #     executor.map(fun, enumerate(shard_filenames))

    for shard in range(0,num_shards):
        tokenize_one(vocab_size, shard_filenames[shard])
    print("Done.")


def pretokenize_target_stories(vocab_size, num_stories, num_stories_Test = 100):
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    # if vocab_size > 0:
    #     # .bin files will be saved into tok{N} directory, create it once here
    #     bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    #     os.makedirs(bin_dir, exist_ok=True)
    #
    # # process all the shards in a process pool
    # fun = partial(process_shard, vocab_size=vocab_size)
    # with ProcessPoolExecutor() as executor:
    #     executor.map(fun, enumerate(shard_filenames))

    # tokenize_one(vocab_size, DATA_CACHE_DIR + "/TinyStories_all_data/data00.json", num_stories, Train = True)
    tokenize_one(vocab_size, DATA_CACHE_DIR + "/TinyStories_all_data/data00.json", num_stories_Test, Train = False)
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
            # bin_dir = os.path.join(DATA_PROCESS_DIR, f"tok{self.vocab_size}")
            bin_dir = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/TinyStories/TinyStories_processing_files/"
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
            # indices = np.where(m)[999]
            indices = [i for i, x in enumerate(self.m == 999) if x]
            indices.insert(0, True)
            num_examples = sum(indices[i] - indices[i-1] - self.max_seq_len for i in range(1,len(indices)))

            self.indices_to_sample_from = []
            for i in range(1, len(indices)):
                self.indices_to_sample_from.extend(range(indices[i-1] + 1, indices[i] - self.max_seq_len))
       
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

            # if chunk[-1] == 999:
            #     start = end + 1
            #     end = start + self.max_seq_len + 1
            #     chunk = torch.from_numpy((m[start:end]).astype(np.int64))

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
        # print("AHAHAHAHAHAHAH")
        # input()
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        # for x, y in dl:
        #     print("In iterbatches for loop")
        #     print(x.shape)
        #     # input()
        #     x = x.to(device, non_blocking=True)
        #     y = y.to(device, non_blocking=True)
        #     yield x, y
        return dl


def find_contexts(vocab_size, context_length, num_stories):

    # how many shards we'll use to find target contexts
    num_shards = 1

    # 1) export a large chunk of text as a single text file tiny.txt
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # retrieve the tokenizer with this list of target tokens
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)

    # A dictionary to keep track of the contexts asociated with this 
    story_context_to_token = []
    total_context_to_token = {}
    total_contexts_encodings = {}
    total_num_context_encountered = 0
    

    for shard in tqdm(shard_filenames[:num_shards]):
        with open(shard, "r") as f:
            data_list = json.load(f)

        # print("Number of stories available are: " + str(len(data_list)))
        # input()
        for data in tqdm(data_list[0:num_stories], desc='Processing'):

            this_story_info = {}

            string_original = data["story"]
            text = string_original.strip()  # get rid of leading/trailing whitespace
            token_encoding = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            # token_strings = enc.encode_as_pieces(text)
            token_strings_list = enc.sp_model.encode_as_pieces(text)

            for token_index in range(context_length, len(token_strings_list)):
                token = token_strings_list[token_index]
                
                context = " ".join(token_strings_list[token_index-context_length:token_index])
                encoding = token_encoding[token_index-context_length:token_index]

                if context in this_story_info.keys():
                    if token in this_story_info[context].keys():
                        this_story_info[context][token] += 1
                    else:
                        this_story_info[context][token] = 1
                else:
                    this_story_info[context] = {}
                    this_story_info[context][token] = 1                
                

                if context in total_context_to_token.keys():
                    if token in total_context_to_token[context].keys():
                        total_context_to_token[context][token] += 1
                    else:
                        total_context_to_token[context][token] = 1
                else:
                    total_context_to_token[context] = {}
                    total_context_to_token[context][token] = 1                



                if context not in total_contexts_encodings.keys():
                    total_contexts_encodings[context] = encoding
                
                total_num_context_encountered += 1
            
            story_context_to_token.append(this_story_info)

    entropy = 0
    for context in total_context_to_token.keys():
        entropy_context = 0
        context_freq = sum(list(total_context_to_token[context].values()))
        for token in total_context_to_token[context].keys():
            entropy_context -= total_context_to_token[context][token] / context_freq * np.log(total_context_to_token[context][token] / context_freq)
        entropy += (context_freq / total_num_context_encountered) * entropy_context


    with open(DATA_PROCESS_DIR + '/story_context_to_token' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'w') as fout:
        json.dump(story_context_to_token, fout)
    
    with open(DATA_PROCESS_DIR + '/total_context_to_token' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'w') as fout:
        json.dump(total_context_to_token, fout)

    with open(DATA_PROCESS_DIR + '/total_contexts_encodings' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'w') as fout:
        json.dump(total_contexts_encodings, fout)

    print("total_contexts_encodings:" + str(len(list(total_contexts_encodings.keys()))))
    print("total_num_context_encountered:" + str(total_num_context_encountered))
    print("Total Entropy:" + str(entropy))
    
    return 


def extract_contexted_stories(vocab_size, target_contexts):

    # how many shards we'll use to find target contexts
    num_shards = 1

    # 1) export a large chunk of text as a single text file tiny.txt
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # retrieve the tokenizer with this list of target tokens
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    target_stories  = []
    
    for shard in tqdm(shard_filenames[:num_shards]):
        with open(shard, "r") as f:
            data_list = json.load(f)

        for data in tqdm(data_list, desc='Processing'):
            string_original = data["story"]
            text = string_original.strip()  # get rid of leading/trailing whitespace
            # token_indices = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            token_strings_list = enc.sp_model.encode_as_pieces(text)

            token_strings_list_string  = " ".join(token_strings_list)
            for context in target_contexts.keys():
                if context in token_strings_list_string:
                    target_stories.append(data)


    print("_" * 100)
    print("Extracted " + str(len(target_contexts)) + " with relevant contexts.")
    print("_" * 100)
    with open(DATA_PROCESS_DIR + "/target_stories.json", 'w') as fout:
      json.dump(target_stories, fout)

    return



def find_target_contexts(num_targets = 100, context_length = 6, vocab_size = 512):

    # with open(DATA_PROCESS_DIR + '/total_context_to_token' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size), 'r') as fout:
    #     total_context_to_token = json.load(fout)
    
    with open(DATA_PROCESS_DIR + '/total_context_to_token' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'r') as j:
        total_context_to_token = json.loads(j.read())
    
    with open(DATA_PROCESS_DIR + '/total_contexts_encodings' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'r') as j:
        total_contexts_encodings = json.load(j)
    
    num_tokens_for_context_dict = {context: sum(list(total_context_to_token[context].values())) for context in total_context_to_token.keys()}
    
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    
    # for i in range(vocab_size):
    #     token = enc.sp_model.IdToPiece(i)
    #     encoding = enc.sp_model.EncodeAsIds(token)
    #     print(f"Token: {token}, Encoding: {encoding}")
    # input()

    # Sort and extract the top 1000 repeated contexts
    sorted_dict = {k: v for k, v in sorted(num_tokens_for_context_dict.items(), key=lambda item: item[1])}
    high_freq_contexts = {k: total_context_to_token[k] for k in list(sorted_dict.keys())[-1000:]}

    # create the supprot matrix for the top 1000
    support_matrix = np.zeros((1000,vocab_size))
    for m in range(0, 1000):
        context = list(high_freq_contexts.keys())[m]
        for v in range(0, vocab_size):
            token = enc.sp_model.IdToPiece(v)
            if token in set(high_freq_contexts[context].keys()):
                support_matrix[m,v] = 1
    
    # Sort the indices by the non diagonal values 
    gram_support_matrix = support_matrix @ support_matrix.T
    gram_support_matrix_nonDiag = gram_support_matrix.copy()
    np.fill_diagonal(gram_support_matrix_nonDiag, 0)
    sorted_indices = np.argsort(gram_support_matrix_nonDiag, axis=None)
    sorted_indices = np.flip(sorted_indices)
    sorted_indices_2d = np.unravel_index(sorted_indices, gram_support_matrix_nonDiag.shape)

    # extract the top #num_targets indices
    list_target_context_indices = set()
    # Print the sorted indices and their corresponding values
    for idx in zip(*sorted_indices_2d):
        if len(list_target_context_indices) < num_targets:
            list_target_context_indices.add(idx[0])
            list_target_context_indices.add(idx[1])
    list_target_context_indices = list(list_target_context_indices)[0:num_targets]

    # Remake the target support matrix
    target_support_matrix = np.zeros((num_targets, vocab_size))
    target_count_matrix = np.zeros((num_targets, vocab_size))
    context_index = 0
    for target_index in list_target_context_indices:
        context = list(high_freq_contexts.keys())[target_index]
        for v in range(0, vocab_size):
            token = enc.sp_model.IdToPiece(v)
            if token in set(high_freq_contexts[context].keys()):
                target_support_matrix[context_index,v] = 1
                target_count_matrix[context_index,v] = high_freq_contexts[context][token]
        context_index += 1

    list_target_contexts = []
    for k_index in list_target_context_indices:
        list_target_contexts.append(list(high_freq_contexts.keys())[k_index])
    
    encoded_target_contexts = np.zeros((num_targets, context_length))
    for index in range(0, num_targets):
        context = list(high_freq_contexts.keys())[list_target_context_indices[index]]
        encoding = total_contexts_encodings[context]
        encoded_target_contexts[index, :] = encoding
    
    
   
    with open(DATA_PROCESS_DIR + '/target_support_matrix' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".npy", 'wb') as fout:
         np.save(fout, target_support_matrix)
    
    with open(DATA_PROCESS_DIR + '/target_count_matrix' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".npy", 'wb') as fout:
         np.save(fout, target_count_matrix)
        
    with open(DATA_PROCESS_DIR + '/encoded_target_contexts' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".npy", 'wb') as fout:
         np.save(fout, encoded_target_contexts)
        
    with open(DATA_PROCESS_DIR + '/list_target_contexts' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'w') as fout:
         json.dump(list_target_contexts, fout)
    
    return 



def find_unique_columns(matrix):
    unique_columns, indices = np.unique(matrix, axis=1, return_index=True)
    return unique_columns[:, np.argsort(indices)]



def find_target_contexts_V2(num_targets = 100, context_length = 6, vocab_size = 512):

    # with open(DATA_PROCESS_DIR + '/total_context_to_token' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size), 'r') as fout:
    #     total_context_to_token = json.load(fout)
    
    with open(DATA_PROCESS_DIR + '/total_context_to_token' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'r') as j:
        total_context_to_token = json.loads(j.read())
    
    with open(DATA_PROCESS_DIR + '/total_contexts_encodings' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'r') as j:
        total_contexts_encodings = json.load(j)
    
    num_tokens_for_context_dict = {context: sum(list(total_context_to_token[context].values())) for context in total_context_to_token.keys()}
    
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    

    num_top_contexts = 1000
    # Sort and extract the top 1000 repeated contexts
    sorted_dict = {k: v for k, v in sorted(num_tokens_for_context_dict.items(), key=lambda item: item[1])}
    high_freq_contexts = {k: total_context_to_token[k] for k in list(sorted_dict.keys())[-num_top_contexts:]}
    
    # create the support matrix for the top 1000
    support_matrix = np.zeros((num_top_contexts,vocab_size))
    for m in range(0, num_top_contexts):
        context = list(high_freq_contexts.keys())[m]
        for v in range(0, vocab_size):
            token = enc.sp_model.IdToPiece(v)
            if token in set(high_freq_contexts[context].keys()):
                support_matrix[m,v] = 1
    
    unique_sup_sets = find_unique_columns(support_matrix)
    unique_sup_sets_counts = {}
    unique_sup_sets_size = {}
    for context_index in range(0, num_top_contexts):
        contxt_sup = support_matrix[context_index,:]

        # print(contxt_sup.shape)
        context_sup_size = sum(contxt_sup)
        contxt_sup = tuple(contxt_sup[row] for row in range(contxt_sup.shape[0]))
        # print(context_sup_size)
        # input()

        if contxt_sup in unique_sup_sets_counts.keys():
            unique_sup_sets_counts[contxt_sup] += 1
        else:
            unique_sup_sets_counts[contxt_sup] = 1
            unique_sup_sets_size[contxt_sup] = context_sup_size
    
    
    
    sorted_unique_sup_sets_counts = {k: v for k, v in sorted(unique_sup_sets_counts.items(), reverse=True, key=lambda item: item[1])}
    sorted_unique_sup_sets_size = {k: unique_sup_sets_size[k] for k in list(unique_sup_sets_counts.keys())}

    # extract the top #num_targets indices
    list_target_context_indices = []
    # Print the sorted indices and their corresponding values
    for support_tuple in list(sorted_unique_sup_sets_counts.keys())[0:10]:
        support_np = np.array(support_tuple)
        for context_index in range(0, num_top_contexts):
            if np.array_equal(support_matrix[context_index, :], support_np):
                list_target_context_indices.append(context_index)
    
    # Remake the target support matrix
    target_support_matrix = np.zeros((len(list_target_context_indices), vocab_size))
    target_count_matrix = np.zeros((len(list_target_context_indices), vocab_size))
    context_index = 0
    for target_index in list_target_context_indices:
        context = list(high_freq_contexts.keys())[target_index]
        for v in range(0, vocab_size):
            token = enc.sp_model.IdToPiece(v)
            if token in set(high_freq_contexts[context].keys()):
                target_support_matrix[context_index,v] = 1
                target_count_matrix[context_index,v] = high_freq_contexts[context][token]
        context_index += 1


    list_target_contexts = []
    for k_index in list_target_context_indices:
        list_target_contexts.append(list(high_freq_contexts.keys())[k_index])
    
    encoded_target_contexts = np.zeros((len(list_target_context_indices), context_length))
    for index in range(0, len(list_target_context_indices)):
        context = list(high_freq_contexts.keys())[list_target_context_indices[index]]
        encoding = total_contexts_encodings[context]
        encoded_target_contexts[index, :] = encoding
    
   
    with open(DATA_PROCESS_DIR + '/target_support_matrix_V2' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".npy", 'wb') as fout:
         np.save(fout, target_support_matrix)
    
    with open(DATA_PROCESS_DIR + '/target_count_matrix_V2' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".npy", 'wb') as fout:
         np.save(fout, target_count_matrix)
        
    with open(DATA_PROCESS_DIR + '/encoded_target_contexts_V2' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".npy", 'wb') as fout:
         np.save(fout, encoded_target_contexts)
        
    with open(DATA_PROCESS_DIR + '/list_target_contexts_V2' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'w') as fout:
         json.dump(list_target_contexts, fout)
    
    return 





def find_shared_support_sets(num_targets = 100, context_length = 6, vocab_size = 512):

    # with open(DATA_PROCESS_DIR + '/total_context_to_token' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size), 'r') as fout:
    #     total_context_to_token = json.load(fout)
    
    with open(DATA_PROCESS_DIR + '/total_context_to_token' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'r') as j:
        total_context_to_token = json.loads(j.read())
    
    with open(DATA_PROCESS_DIR + '/total_contexts_encodings' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'r') as j:
        total_contexts_encodings = json.load(j)
    
    num_tokens_for_context_dict = {context: sum(list(total_context_to_token[context].values())) for context in total_context_to_token.keys()}
    
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    
    

    with open(DATA_PROCESS_DIR + '/support_set_dicts' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'r') as fout:
        support_set_dicts = json.load(fout)
    with open(DATA_PROCESS_DIR + '/keys_token_list' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".json", 'r') as fout:
        keys_token_list = json.load(fout)
    

    # count_support_set = {sup_set:len(set(support_set_dicts[sup_set])) for sup_set in list(support_set_dicts.keys())}
    count_support_set = {sup_set: sup_set.count("1") for sup_set in list(support_set_dicts.keys())}
    sorted_count_support_set = {k: v for k, v in sorted(count_support_set.items(), key=lambda item: item[1])}
    
    num_targets = 100
    target_shared_support_matrix = np.zeros((num_targets, vocab_size))
    target_shared_count_matrix = np.zeros((num_targets, vocab_size))

    num_samples_per_context = 10
    shared_support_set_sampled_contexts = []
    example_counter = 0
    for support_set in list(sorted_count_support_set.keys())[-100000:]:
        support_set_contexts = support_set_dicts[support_set]
      
        if len(support_set_contexts) >= 10:
            counter = 0
            i = 0
            while counter < 10:
                
                if sum(list(total_context_to_token[keys_token_list[support_set_contexts[i]]].values())) >= 1:
                    shared_support_set_sampled_contexts.append(total_contexts_encodings[keys_token_list[support_set_contexts[i]]])
                    counter += 1


                    for v in range(0, vocab_size):
                        token = enc.sp_model.IdToPiece(v)
                        if token in set(total_context_to_token[keys_token_list[support_set_contexts[i]]].keys()):
                            target_shared_support_matrix[example_counter,v] = 1
                            target_shared_count_matrix[example_counter,v] = total_context_to_token[keys_token_list[support_set_contexts[i]]][token]
                    
                    example_counter += 1
                

                i += 1
        
        if len(shared_support_set_sampled_contexts) >= 100:
            break
                
    
    shared_support_set_sampled_contexts = np.array(shared_support_set_sampled_contexts)
    with open(DATA_PROCESS_DIR + '/shared_support_set_sampled_contexts' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".npy", 'wb') as fout:
        np.save(fout, shared_support_set_sampled_contexts)
    
    with open(DATA_PROCESS_DIR + '/target_shared_support_matrix' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".npy", 'wb') as fout:
        np.save(fout, target_shared_support_matrix)
    
    with open(DATA_PROCESS_DIR + '/target_shared_count_matrix' + "_contextLenght_" + str(context_length) + "_vocabSize_" + str(vocab_size) + ".npy", 'wb') as fout:
        np.save(fout, target_shared_count_matrix)

    return 


            
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
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab", "process", "target_contexts", "shared_contexts"])
    parser.add_argument("--vocab_size", type=int, default=64, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--num_stories", type=int, default=1000, help="Number of stories to use")
    parser.add_argument("--context_length", type=int, default=6, help="Number of stories to use")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size, target_tokens = [], num_stories = args.num_stories)
        pretokenize_target_stories(vocab_size=args.vocab_size, num_stories = args.num_stories, num_stories_Test = 100)
    elif args.stage == "target_contexts":
        # find_target_contexts(num_targets = 100, context_length = args.context_length, vocab_size = args.vocab_size)
        find_target_contexts_V2(num_targets = 100, context_length = args.context_length, vocab_size = args.vocab_size)
    elif args.stage == "shared_contexts":
        find_shared_support_sets(num_targets = 100, context_length = args.context_length, vocab_size = args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
