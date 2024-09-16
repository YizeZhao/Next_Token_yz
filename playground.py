import sentencepiece as spm
from tokenizer import Tokenizer
import string
import numpy as np

S = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
ST = S.T

print(f"S @ ST: {S @ ST}")
print(f"ST @ S: {ST @ S}")
print(f"S @ ST @ S @ ST: {S @ ST @ S @ ST}")
print(f"ST @ S @ ST @ S: {ST @ S @ ST @ S}")

# P = np.load(f'./results_tiny_extract_m404_2/P_UFM.npy')
#
# print(P.shape)
exit(0)
model_file='./tokenizer/tok_tiny_2000lines_char.model'
vocab_file = './tokenizer/tok152.vocab'
sp = spm.SentencePieceProcessor(model_file=model_file)
vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
for i in range(sp.get_piece_size()):
    print(i, vocabs[i])

# with open('/Users/yizezhao/Documents/Datasets/tok2048/data00.bin', mode='rb') as file: # b is important -> binary
#     fileContent = file.read()
#
# print(fileContent)

# data_file = "./verysmallset.txt"
# t = Tokenizer(model_file)
# with open(data_file, "r") as f:
#     text = f.read().lower().translate(str.maketrans('', '', string.punctuation))
#     text = text.replace('\n', ' .\n')
#     encoded = t.encode(text,bos=True, eos=True)
#
# tokens = [sp.IdToPiece(id_) for id_ in encoded]
# print(tokens)
#

def spm_export_vocab(model_file, vocab_file = None):
    vocab_dict = {}
    if vocab_file is None:
        vocab_file = model_file.replace(".model", ".vocab")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    vocabs = [sp.IdToPiece(id) for id in range(sp.GetPieceSize())]
    with open(vocab_file, "w") as vfile:
        for v in vocabs:
            id = sp.PieceToId(v)
            vocab_dict[v] = sp.GetScore(id)
            vfile.write(f'{v}\t{sp.GetScore(id)}\n')

    return vocab_dict

# vocab_dict_2000 = spm_export_vocab(model_file, vocab_file = None)
# print('hello')