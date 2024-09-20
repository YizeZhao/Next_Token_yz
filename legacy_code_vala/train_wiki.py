"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task
from export import model_export

import torch._dynamo
torch._dynamo.config.suppress_errors = True
from tqdm import tqdm

import time 
import argparse


import json 
from tinystories import get_tokenizer_model_path
from tokenizer import Tokenizer
import wandb
from loggers_full import * 
import psutil
from tinystories import PretokDataset



parser = argparse.ArgumentParser('arguments for training')
parser.add_argument("--version", type=int, default=1)
parser.add_argument("--num_stories", type=int, default=1000)

parser.add_argument("--vocab_size", type=int, default=64)
parser.add_argument('--vocab_size_index', type=int, default=0, help='dimension of last layer')

parser.add_argument("--max_seq_len", type=int, default=6)
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
parser.add_argument("--pos_enc", type=str, default="off", help='Positional Encoding')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--epochs', type=int, default=600, help='number of training epochs')
parser.add_argument('--dim', type=int, default=128, help='dimension of last layer')

parser.add_argument("--AR_training", action='store_true', help='Auto Regressive Training')
parser.add_argument("--Debug", action='store_true', help='Use when debugging')

parser.add_argument('--n_layers', type=int, default=12, help='dimension of last layer')
args = parser.parse_args()


def main():

    vocab_size_list = [128, 256, 512, 1024, 2048]
    args.vocab_size = vocab_size_list[args.vocab_size_index]


    version_log = "logs_V" + str(args.version)
    # -----------------------------------------------------------------------------
    # I/O
    out_dir = "out_2layer_pos_off"
    eval_only = False  # if True, script exits right after the first eval
    init_from = "scratch"  # 'scratch' or 'resume'
    # wandb logging
    wandb_log = True  # disabled by default
    # data
    batch_size = args.batch_size  # if gradient_accumulation_steps > 1, this is the micro-batch size
    max_seq_len = args.max_seq_len
    vocab_source = "custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
    vocab_size = args.vocab_size # the Llama 2 tokenizer has 32K tokens
    num_stories = args.num_stories
    # model
    dim = args.dim
    n_layers = args.n_layers
    n_heads = 6
    n_kv_heads = 6
    multiple_of = 32
    dropout = 0.0
    pos_enc = args.pos_enc
    # adamw optimizer
    # gradient_accumulation_steps = 1  # used to simulate larger batch sizes
    learning_rate = args.learning_rate  # max learning rate
    weight_decay = args.weight_decay
    beta1 = 0.9
    beta2 = 0.95
    # grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 5  # how many steps to warm up for
    # system
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = "float16"  # float32|bfloat16|float16
    compile = True  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    num_epochs = args.epochs


    context_repeat_threshold = 3
    DATA_PROCESS_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/TinyStories/TinyStories_processing_files"

    # validating checks
    assert vocab_source in ["llama2", "custom"]
    assert vocab_source == "custom" or vocab_size == 32000, "The vocab from Meta has 32K tokens"

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

    tokens_per_iter =  ddp_world_size * batch_size * max_seq_len
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]


    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pos_enc=pos_enc,
        AR_training = args.AR_training,
    )  # start with model_args from command line
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    # model= torch.nn.DataParallel(model)
    model.to(device)
    print(model)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == "resume" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0


    # logging
    if wandb_log and master_process:
        if args.Debug:
            wandb.init(
                # set the wandb project where this run will be logged
                project="NTP_Deepnet_V2", name="DEBUG_correct_V" + str(args.version) + "_" + str(args.n_layers) + "L_v" + str(args.vocab_size) + "_s" + str(args.num_stories) + "_d" + str(args.dim) +"_lr" + str(args.learning_rate) + "_total_e" + str(args.epochs) + "_exp_lrdecay" + "_ARtrain_" + str(args.AR_training)
            )
        else:
            wandb.init(
                # set the wandb project where this run will be logged
                project="NTP_Deepnet_V2", name="correct_V" + str(args.version) + "_" + str(args.n_layers) + "L_v" + str(args.vocab_size) + "_s" + str(args.num_stories) + "_d" + str(args.dim) +"_lr" + str(args.learning_rate) + "_total_e" + str(args.epochs) + "_exp_lrdecay" + "_ARtrain_" + str(args.AR_training)
            )

    # training loop
    # X, Y = next(train_batch_iter)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    # raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0


    context_logger = logger()

    iteration_counter = 0
    log_iteration = 0

    # fixing some hyperparams to sensible defaults
    lr_decay_iters = num_epochs  # should be ~= max_iters per Chinchilla
    min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)


    ds = PretokDataset("train", max_seq_len, vocab_size, vocab_source, num_stories, AR_training = args.AR_training)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, pin_memory=True, num_workers=0
    )


    if args.AR_training:
        entropy = 0
        entropy_pos = {}
        pos_context_count = {}
        for pos in range(1, max_seq_len + 1):
            entropy_pos[pos] = 0
            pos_context_count[pos] = 0


        context_dict = {}
        num_datapoints = 0
        for batch_idx, (X, Y) in enumerate(dl):
            for index in range(0, Y.shape[0]):
                x = X[index, :]
                y = Y[index, :]

                for i in range(1, max_seq_len + 1):
                    context = x[:i]
                    token = y[i - 1]

                    pos_context_count[len(context)] += 1

                    context = "_".join([str(n) for n in context.tolist()])
                    token = token.item()

                    if context in context_dict.keys():
                        if token in context_dict[context].keys():
                            context_dict[context][token] += 1
                        else:
                            context_dict[context][token] = 1
                    else:
                        context_dict[context] = {}
                        context_dict[context][token] = 1

                    num_datapoints += 1

        for context in context_dict.keys():
            entropy_context = 0
            context_freq = sum(list(context_dict[context].values()))

            len_context = len(context.split("_"))
            
            for token in context_dict[context].keys():
                entropy_context -= context_dict[context][token] / context_freq * np.log(context_dict[context][token] / context_freq)
            
            entropy_pos[len_context] += (context_freq / pos_context_count[len_context]) * entropy_context

            entropy += (context_freq / num_datapoints) * entropy_context

        entropy = entropy * max_seq_len
        
        print("*" * 100)
        print("Number of contexts: " + str(len(list(context_dict.keys()))) )
        print("entropy: " + str(entropy))
        print("entropy_pos: " + str(entropy_pos))
        print("ave entropies: " + str(sum(list(entropy_pos.values())) / max_seq_len))
        print("*" * 100)
    else:
        context_dict = {}
        num_datapoints = 0
        for batch_idx, (X, Y) in enumerate(dl):
            for index in range(0, Y.shape[0]):
                x = X[index, :]
                y = Y[index]

                context = "_".join([str(n) for n in x.tolist()])
                token = y.item()
                
                if context not in context_dict.keys():
                    context_dict[context] = {}

                if token in context_dict[context]:
                    context_dict[context][token] += 1
                else:
                    context_dict[context][token] = 1
                
                num_datapoints += 1

        entropy = 0
        for context in context_dict.keys():
            entropy_context = 0
            context_freq = sum(list(context_dict[context].values()))
            for token in context_dict[context].keys():
                entropy_context -= context_dict[context][token] / context_freq * np.log(context_dict[context][token] / context_freq)
            entropy += (context_freq / num_datapoints) * entropy_context

        print("*" * 100)
        print("Number of contexts: " + str(len(list(context_dict.keys()))) )
        print("entropy: " + str(entropy))
        print("*" * 100)


    total_loss_list = []
    total_loss_per_batch_ave_list = []

    WB_epochs_list = []
    WB_loss_list = []
    WB_lr_list = []


    if args.Debug:
        save_path = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/train_files/DEBUG_dim" + str(dim) + "_vocabSize" + str(vocab_size) + "_numStories" + str(args.num_stories) + "_ARtrain_" + str(args.AR_training)
    else:
        save_path = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_V2/train_files/" + version_log + "/dim" + str(dim) + "_vocabSize" + str(vocab_size) + "_numStories" + str(args.num_stories) + "_ARtrain_" + str(args.AR_training)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), save_path + "/model_it" + "0" + ".pth")

    lr = learning_rate
    for epoch in range(0, num_epochs):

        print("_" * 100)
        print("Epoch " + str(epoch))
        print("_" * 100)

        epoch_time = time.time()
   

        lr = get_lr(epoch) if decay_lr else learning_rate
        # lr = get_lr(epoch, lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        num_loss_calc = 0
        total_loss = 0
        total_loss_per_batch_ave = 0
        # while True:
        num_batches = 0

        print(ds.indices_to_sample_from[0:100])


        loss_per_pos = {}
        for pos in range(0, max_seq_len):
            loss_per_pos[pos] = [0 for i in range(0, num_epochs)]

        num_datapoints = 0
        for batch_idx, (X, Y) in tqdm(enumerate(dl)):


            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            
            if iter_num == 0 and eval_only:
                break
            
            if args.AR_training:
                logits, h, loss_per_pos_batch = model(X, Y)

                for pos in range(0, max_seq_len):
                    loss_per_pos[pos][epoch] += loss_per_pos_batch[pos]
            else:
                logits, h = model(X, Y)
            
            loss = model.last_loss
            iteration_counter += 1

            total_loss += loss.item()
            total_loss_per_batch_ave += loss.item()

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            

            # clip the gradient
            # if grad_clip != 0.0:
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            iter_num += 1
            local_iter_num += 1

            num_loss_calc += 1

            num_datapoints += Y.shape[0]

            num_batches += 1

          
        if args.AR_training:
            for pos in range(0, max_seq_len):
                loss_per_pos[pos][epoch] /= num_datapoints
        

        total_loss_list.append(total_loss / num_datapoints)
        total_loss_per_batch_ave_list.append(total_loss_per_batch_ave / num_batches)

        WB_epochs_list.append(epoch)
        WB_loss_list.append(total_loss / num_datapoints)
        WB_lr_list.append(lr)

        weight_norm = torch.norm(model.output.weight, p = 2).detach().item()
        
        if args.AR_training:
            loss_per_pos_minus_entropy = []
            for pos in range(1, max_seq_len + 1):
                loss_per_pos_minus_entropy.append(loss_per_pos[pos - 1][epoch] - entropy_pos[pos])

        if wandb_log:

            if args.AR_training:
                try:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "loss/train": total_loss / num_datapoints - entropy ,
                            "lr": lr,
                            "W norm": weight_norm,
                            "loss per pos 1": loss_per_pos_minus_entropy[0],
                            "loss per pos 2": loss_per_pos_minus_entropy[1],
                            "loss per pos 3": loss_per_pos_minus_entropy[2],
                            "loss per pos 4": loss_per_pos_minus_entropy[3],
                            "loss per pos 5": loss_per_pos_minus_entropy[4],
                            "loss per pos 6": loss_per_pos_minus_entropy[5],
                        }, step = epoch
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            else:
                try:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "loss/train": total_loss / num_datapoints - entropy,
                            "lr": lr,
                            "W norm": weight_norm,
                        }, step = epoch
                    )

                except Exception as e:
                    print(f"logging to wandb failed: {e}")

        
        epoch_time = time.time() - epoch_time
        print("*" * 100)
        print("This epoch took " + str(epoch_time / 60) + " minutes.")
        print("*" * 100)
        print("num_datapoints: " + str(num_datapoints))
        print("Logging the info ...")
        model.eval()      

        log_iteration += 1

        if log_iteration % 10 == 0 or log_iteration in [0, 1, 3, 5, num_epochs - 5, num_epochs - 2, num_epochs]:
            torch.save(model.state_dict(), save_path +  "/model_it" + str(log_iteration) + ".pth")
            torch.save(optimizer.state_dict(), save_path + "/optimizer_it" + str(log_iteration) + ".pth")
            with open(save_path + "/loss_list.npy", 'wb') as f:
                np.save(f, total_loss_list)
            with open(save_path + "/loss_per_batch_ave_list.npy", 'wb') as f:
                np.save(f, total_loss_per_batch_ave_list)
            
            if args.AR_training:
                with open(save_path + "/loss_per_pos.json", 'w') as f:
                    json.dump(loss_per_pos, f)
        

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        print("GPU Memory Allocated:", torch.cuda.memory_allocated(device) / (1024 ** 3), "GB")
        print("Total Loss : " + str(total_loss))
        print("Total Loss (per batch average): " + str(total_loss_per_batch_ave / num_batches))
        print("Average Loss : " + str(total_loss / 521404))
        print("Learning rate: " + str(lr))
        print("Done. ")

        if ddp:
            destroy_process_group()


if __name__ == '__main__':
    main()