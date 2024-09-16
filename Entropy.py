'''
1 compute empirical entropy of a text body
'''

import numpy as np
from itertools import tee, islice
from collections import Counter
import re
import sentencepiece as spm
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import json


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


syn_sub = {
    "liked":["likes", "loved", "loves", "enjoyed", "enjoyes", "prefers", "preferred", "favored", "favors", "admires", "admired", "adores", "adored", "cherishes", "cherished"],
    "loved":["likes", "liked", "loves", "enjoyed", "enjoyes", "prefers", "preferred", "favored", "favors", "admires", "admired", "adores", "adored", "cherishes", "cherished"],
    "little":["young", "small", "smart", "cute", "bright", "playful", "tiny", "short", "sweet", "adorable", "charming", "delightful","youthful", "smallish", "undersized"\
              "lovable", "darling", "precious", "enchanting", "petite"],
    "day": ["night", "morning", "afternoon", "evening", "dawn", "dusk", "noon", "midnight", "sunrise", "sunset"],
    "lots": ["alot", "many", "much", "plenty"],
    "lily": ["amy", "lucy", "anna", "sarah", "lilly", "lucie", "annie", "sara"],
    "tim": ["tom", "bob", "Sam", "timmy", "timothy", "timmothy", "timmo", "timothy", "timmothy", "timmo", "timothy", "timmothy", "timmo", "timothy", "timmothy", "timmo"],
    "timmy": ["tom", "bob", "Sam", "tim", "timothy", "timmothy", "timmo", "timothy", "timmothy", "timmo", "timothy", "timmothy", "timmo", "timothy", "timmothy", "timmo"],
    "big": ["massive", "huge", "grand", "Enormous", "giant", "large", "great", "immense", "substantial", "tremendous", "vast", "colossal", "mammoth", "monumental", "mountainous", "towering", "titanic", "elephantine", \
            "gargantuan", "gigantic", "humongous", "jumbo", "king-size", "king-sized "],
    "named": ["called", "titled", "entitled", "designated", "denominated", "christened", "labelled", "dubbed", "termed", "identified", "specified", "tagged",\
              "nicknamed", "branded", "styled", "cited", "referred", "denoted", "entitled", "labelled", "designated", "christened", "dubbed", "tagged", "termed", "styled"],
    "boy": ["man", "male", "guy", "dude", "gentleman", "fellow", "chap", "bloke", "youth", "youngster", "teen", "teenager", "adolescent", "junior", "son"],
    "girl": ["lady", "female", "maiden", "lass", "lassie", "damsel", "miss", "babe", "bairn", "child", "kid"]
}

verb_sub = {
    "be":"is",
    "have":"has",
    "go":"goes",
    "name":"named"
}

skip = ["time", "named"]

# Save dictionary to a file
def save_dict_to_file(filename, dictionary):
    with open(filename, 'w') as f:
        json.dump(dictionary, f)

# Load dictionary from a file
def load_dict_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def find_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)



def substitute_words(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Tag each word with its part of speech
    tagged_words = pos_tag(words)

    # Initialize a list to store the substituted sentences
    substituted_sentences = []

    for i, (word, tag) in enumerate(tagged_words[:-2]):
        # if tag.startswith('JJ') or tag.startswith('NN') or tag.startswith('VB'):
        synonyms = None
        if word in skip:
            continue
        elif word in syn_sub.keys():
            synonyms = syn_sub[word]
        elif tag.startswith('JJ') or tag.startswith('NN'):
            # Find synonyms of the adjective
            synonyms_all = find_synonyms(word)
            synonyms = synonyms_all[:min(10, len(synonyms_all))]

        if synonyms:
            # Choose one synonym to replace the original adjective
            # new_adjective = synonyms  # You can choose a random synonym here instead

            # Create a new sentence with the adjective substituted
            for new_adjective in synonyms:
                if '_' in new_adjective:
                    continue
                new_sentence = ' '.join([new_adjective if j == i else w for j, w in enumerate(words)])
                substituted_sentences.append(new_sentence)

    return substituted_sentences

def present_tense_third_person_singular(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Tag each word with its part of speech
    tagged_words = pos_tag(words)

    # Initialize WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Find the verb in the sentence and change it to present tense third-person singular form
    for word, tag in tagged_words:
        if word in skip:
            continue
        if tag.startswith('VB'):  # Check if the word is a verb
            # Lemmatize the verb to its base form
            base_form = lemmatizer.lemmatize(word, pos='v')
            # Change the verb to present tense third-person singular form
            if base_form in verb_sub.keys():
                return sentence.replace(word, verb_sub[base_form])
            if base_form == word:
                # If the base form is the same as the word itself, it's likely already in present tense third-person singular
                continue
            elif base_form.endswith('s') or base_form.endswith('sh') or base_form.endswith('ch') or base_form.endswith(
                    'x') or base_form.endswith('z'):
                # If the base form ends with these suffixes, just add 'es'
                return sentence.replace(word, base_form + 'es')
            elif base_form.endswith('y') and len(base_form) > 1 and base_form[-2] not in 'aeiou':
                # If the base form ends with 'y' preceded by a consonant, replace 'y' with 'ies'
                return sentence.replace(word, base_form[:-1] + 'ies')
            else:
                # Otherwise, just add 's'
                return sentence.replace(word, base_form + 's')

    # If no verb is found, return the original sentence
    return None

def ngrams(lst, n):
  tlst = lst
  while True:
    a, b = tee(tlst)
    l = tuple(islice(a, n))
    if len(l) == n:
      yield l
      next(b)
      tlst = b
    else:
      break

def write2txt(ctx_dict, txt_path):
    with open(txt_path, 'w') as f:
        for ctx_k, ctx_v in ctx_dict.items():
            for nt in ctx_v['s_cnt_str']:
                ctx_nt = ctx_k + " " + nt
                ctx_nt = ctx_nt.replace("▁", "")
                ctx_nt = ctx_nt.replace(" +", " ")
                f.write(ctx_nt + '\n')
    return

def cherry_pick_by_freq(ctx_dict, freq_thres = 2):
    # find the context
    filtered_ctx_dict = {}
    for ctx_k, ctx_v in ctx_dict.items():
        s_cnt = ctx_v['s_cnt_ids']
        s_str_cnt = ctx_v['s_cnt_str']
        if len(list(s_str_cnt)) >= 8:
            filtered_s_cnt = {key: count for key, count in s_cnt.items() if count >= freq_thres}
            filtered_s_str = {key: count for key, count in s_str_cnt.items() if count >= freq_thres}
        else:
            filtered_s_cnt = s_cnt
            filtered_s_str = s_str_cnt
        new_s_n = len(list(filtered_s_cnt.keys()))
        if new_s_n > 0:
            filtered_ctx_dict[ctx_k] = ctx_dict[ctx_k]
            filtered_ctx_dict[ctx_k]['s_n'] = new_s_n
            filtered_ctx_dict[ctx_k]['s_cnt_ids'] = filtered_s_cnt
            filtered_ctx_dict[ctx_k]['s_cnt_str'] = filtered_s_str

    m,v, nts_uniq = count_m_v(filtered_ctx_dict, return_np_unique=True)
    h = compute_entropy(filtered_ctx_dict)
    print(f"new context set (filtered by frequency):: m = {m} | v = {v} | entropy = {h}")

    # filter again but keep samples in selected v
    filtered_ctx_dict_2 = {}
    for ctx_k, ctx_v in ctx_dict.items():
        s_cnt = ctx_v['s_cnt_ids']
        filtered_s_cnt = {key: count for key, count in s_cnt.items() if key in nts_uniq}
        new_s_n = len(list(filtered_s_cnt.keys()))
        if new_s_n > 0:
            filtered_ctx_dict_2[ctx_k] = ctx_dict[ctx_k]
            filtered_ctx_dict_2[ctx_k]['s_n'] = new_s_n
            filtered_ctx_dict_2[ctx_k]['s_cnt_ids'] = filtered_s_cnt

    m,v, nts_uniq = count_m_v(filtered_ctx_dict_2, return_np_unique=True)
    h = compute_entropy(filtered_ctx_dict_2)
    print(f"new context set (filtered by frequency):: m = {m} | v = {v} | entropy = {h}")

    get_support_distribution(filtered_ctx_dict_2)

    return filtered_ctx_dict_2

def cherry_pick_by_entropy(ctx_dict, entropy_dict):
    # find the context
    filtered_ctx_dict = {}
    for ctx_k, ctx_v in entropy_dict.items():
        filtered_ctx_dict[ctx_k] = ctx_dict[ctx_k]

    m, v = count_m_v(filtered_ctx_dict)
    print(f"new context set (filtered by entropy): m = {m} | v = {v}")
    return filtered_ctx_dict

def count_m_v(ctx_dict, return_np_unique = False):
    nts = []
    m = len(list(ctx_dict.keys()))
    for ctx_k, ctx_v in ctx_dict.items():
        nts.extend(list(ctx_v['s_cnt_ids'].keys()))

    nts_unique = np.unique(nts)
    v = len(nts_unique)

    if return_np_unique:
        return m,v,nts_unique
    return m,v

def compute_entropy(ctx_dict):
    emp_entropy = 0
    freq_sum_all = 0
    for ctx_k, ctx_v in ctx_dict.items():
        ctx_v_repeats = np.array(list(ctx_v['s_cnt_ids'].values()))
        freq_sum = ctx_v_repeats.sum()
        ctx_v_pr = ctx_v_repeats/freq_sum
        ctx_entropy = - np.sum(ctx_v_pr * np.log(ctx_v_pr)) * freq_sum

        freq_sum_all += freq_sum
        emp_entropy += ctx_entropy

    emp_entropy = emp_entropy/freq_sum_all
    print(f"Empirical entropy: {emp_entropy}")
    return emp_entropy

def get_support_distribution(ctx_dict):
    m, v, nts = count_m_v(ctx_dict, return_np_unique=True)
    temp_id2tok = np.unique(nts)
    tok2temp_id = {}
    for temp_id in range(len(temp_id2tok)):
        tok = temp_id2tok[temp_id]
        tok2temp_id[tok] = temp_id
    supports = np.zeros((m,v))
    s_ns = []
    m_ = 0
    for ctx_k, ctx_v in ctx_dict.items():
        s_ns.append(ctx_v['s_n'])
        # print(f"ctx - {ctx_k} | support - {ctx_v['s_cnt_str']}")
        for tok in list(ctx_v['s_cnt_ids'].keys()):
            supports[m_][tok2temp_id[tok]] = 1
        m_ += 1
    s_ns_mean = np.array(s_ns).mean()
    print(f"average support length: {s_ns_mean}")

    nrows, ncols = 2, 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=100)
    im1 = axs[0].imshow(supports, cmap='hot')
    axs[0].set_title('Support')

    im2 = axs[1].imshow(supports @ supports.T, cmap='hot', vmin=0, vmax=7)
    axs[1].set_title('Support')


    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    for im_ in [im1, im2]:
        plt.colorbar(im_)

    # plt.savefig(f'./figures/Suppot_sampled_tinystories.png')
    plt.show()



def getContextEntropy(tok_list, T, tokenizer=None, n=50):
    # tok_list = np.memmap(tok_file, dtype=np.uint16, mode="r")
    ngrams_all = ngrams(lst=tok_list, n=T-1)
    ngrams_cnt = Counter(ngrams_all)
    ngrams_len = 0
    uniq_ctx_len = len(ngrams_cnt)
    most_common = ngrams_cnt.most_common(n)
    # print(uniq_ctx_len)
    # ctx_supports,ctx_pr = {},{}
    emp_entropy = 0
    larger_one_cnt = 0
    ctx_dict = {}
    entropy_dict = {}
    nts = []
    for t_, freq_ in most_common:
        tok_ = ctx =  np.array(t_)
        if tokenizer is not None:
            ctx = [sp.id_to_piece(int(id)) for id in tok_]
            ctx = " ".join(ctx)
        ngrams_len += freq_
        if freq_ <= 1:
            continue
        if any([0,1,2]) in tok_:
            continue
        idxes = np.where(np.all(np.lib.stride_tricks.sliding_window_view(tok_list[:-T], tok_.shape) == tok_,
                        axis=-1))[0]

        next_tokens = tok_list[idxes + T-1]
        next_token_ctx = [sp.id_to_piece(int(id)) for id in next_tokens]
        next_token_ctx_str = " ".join(next_token_ctx)
        next_tokens_cnt = Counter(next_tokens)
        next_tokens_ctx_cnt = Counter(next_token_ctx)
        s = len(next_tokens_ctx_cnt.values())
        if s<=1:
            continue
        larger_one_cnt += 1
        nts.extend(next_tokens)
        ctx_dict[ctx] = {}
        ctx_dict[ctx]['s_n'] = s
        ctx_dict[ctx]['ctx_ids'] = tok_
        ctx_dict[ctx]['s_cnt_str'] = next_tokens_ctx_cnt
        ctx_dict[ctx]['s_cnt_ids'] = next_tokens_cnt
        # print(f"ctx - {ctx} s - {s}  - {next_tokens_ctx_cnt}")

        support,pr = [],[]
        if s == 1:
            continue
        # entropy:
        for s_, freq_s_ in next_tokens_ctx_cnt.items():
            support.append(s_)
            pr.append(freq_s_)
        support, pr = np.array(support), np.array(pr)
        freq_sum = pr.sum()
        # assert freq_sum == freq_
        pr = pr/freq_sum

        ctx_entropy = - np.sum(pr * np.log(pr)) * freq_sum
        entropy_dict[ctx] = ctx_entropy
        emp_entropy += ctx_entropy


    # analyze(ctx_dict)
    m = len(list(ctx_dict.keys()))
    v = len(set(nts))
    # emp_entropy = emp_entropy/ngrams_len
    print(f" initial m = {m} | v = {v} | emp entropy - {emp_entropy/ngrams_len}")
    # print(larger_one_cnt)

    filtered_ctx_dict_2 = cherry_pick_by_freq(ctx_dict=ctx_dict, freq_thres=2)

    sorted_entropy_list = sorted(entropy_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_entropy_list_sel = sorted_entropy_list[:200]
    sorted_entropy_dict_sel = dict(sorted_entropy_list_sel)
    # cherry_pick_by_entropy(ctx_dict=ctx_dict, entropy_dict=sorted_entropy_dict_sel)

    return filtered_ctx_dict_2

def fixedTentropy(tok_list, T, tokenizer=None):
    # tok_list = np.memmap(tok_file, dtype=np.uint16, mode="r")
    if T > 1:
        ngrams_all = ngrams(lst=tok_list, n=T-1)
        ngrams_cnt = Counter(ngrams_all)
    else:
        ngrams_cnt = Counter(tok_list)
    ngrams_len = 0
    # print(uniq_ctx_len)
    # ctx_supports,ctx_pr = {},{}
    s_list = []
    emp_entropy = 0
    larger_one_cnt = 0
    for (t_, freq_) in ngrams_cnt.items():
        tok_ = np.array(t_)
        ngrams_len += freq_
        if freq_ > 1:
            idxes = np.where(np.all(np.lib.stride_tricks.sliding_window_view(tok_list[:-T], tok_.shape) == tok_,
                                axis=-1))[0]

            next_tokens = tok_list[idxes + T - 1]
            next_tokens_cnt = Counter(next_tokens)
            s = len(next_tokens_cnt.values())
            larger_one_cnt += 1
            # print(f"ctx - {tok_} freq - {freq_} support - {next_tokens_cnt}")
            s_list.append(s)
            support,pr = [],[]
            if s == 1:
                continue
            # entropy:
            for s_, freq_s_ in next_tokens_cnt.items():
                support.append(s_)
                pr.append(freq_s_)
            support, pr = np.array(support), np.array(pr)
            freq_sum = pr.sum()
            # assert freq_sum == freq_
            pr = pr/freq_sum

            emp_entropy -= np.sum(pr * np.log(pr)) * freq_sum

    # emp_entropy = emp_entropy/ngrams_len
    avg_s = np.array(s_list).mean()
    print(f"total ctx - {ngrams_len} emp entropy - {emp_entropy/ngrams_len}")
    print(f"average support length: {avg_s}")
    # print(larger_one_cnt)
    return emp_entropy/ngrams_len
    # return emp_entropy

def expand_m(ctx_dict):
    ctx_dict_expand = {}
    for ctx_k, ctx_v in ctx_dict.items():
        # ctx_dict_expand[ctx_k] = ctx_v
        ctx_str = ctx_k.replace(" ▁", " ")
        ctx_str = ctx_str.replace("▁", "")
        print(f"ctx - {ctx_str} | support - {ctx_v['s_cnt_str']}")
        ctx_dict_expand[ctx_str] = ctx_v
        verb_simiar = present_tense_third_person_singular(ctx_str)

        if not verb_simiar is None:
            # verb_simiar = verb_simiar.replace(" ", " ▁")
            ctx_dict_expand[verb_simiar] = ctx_v
            print(f"ctx - {verb_simiar} | support - {ctx_v['s_cnt_str']}")
        other_simiars = substitute_words(ctx_str)
        for other_simiar in other_simiars:
            # other_simiar = other_simiar.replace(" +", " ▁")
            ctx_dict_expand[other_simiar] = ctx_v
            print(f"ctx - {other_simiar} | support - {ctx_v['s_cnt_str']}")

    m,v = count_m_v(ctx_dict_expand)
    h = compute_entropy(ctx_dict_expand)
    print(f" after expand m = {m} | v = {v} | entropy {h}" )

    return ctx_dict_expand

def ar_entropy():
    return

def test_word_ngrams(pretok_file):
    with open(pretok_file, "r") as f:
        pretok_text = f.read()

    words = re.findall("\w+", pretok_text)
    cnt = Counter(ngrams(words, 5))
    most_common_50 = cnt.most_common(50)
    print(cnt)

if __name__ == '__main__':
    # # Test the function
    sentence = "He ate apples and oranges."
    # present_tense = present_tense_third_person_singular(sentence)
    # print("Present tense:", present_tense)
    #
    # sentence = "The big brown dog jumps over the lazy fox."
    # substituted_sentences = substitute_adjective(sentence)
    # print("Original sentence:", sentence)
    # print("Substituted sentences:")
    # for s in substituted_sentences:
    #     print(s)
    #
    # exit(0)
    # pretok_file = "./data/tiny_2000lines_pretok.txt"
    # tok_file = "./data/tiny_2000lines_pretok_word.bin"
    #
    # spm_model_path = "./tokenizer/tok_tiny_2000lines_word.model"
    # save_json = "./data/expanded_ctx_dict_2.json"
    # save_txt = "./data/tiny_extract_test.txt"
    # sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    #
    #
    # tok_list = np.memmap(tok_file, dtype=np.uint16, mode="r")
    # filtered_ctx_dict_2 = getContextEntropy(tok_list, 6, tokenizer=sp, n=50)
    # expanded_ctx_dict_2 = expand_m(filtered_ctx_dict_2)
    # get_support_distribution(expanded_ctx_dict_2)
    # # save_dict_to_file(save_json, expanded_ctx_dict_2)
    # write2txt(expanded_ctx_dict_2, save_txt)

    # tok_file = "./data/tiny_2000lines_pretok_char.bin"
    # spm_model_path = "./tokenizer/tok_tiny_2000lines_char.model"

    # dataset_name = "tiny100lines"
    # # dataset_name = "tiny_2000lines"
    # tok_file = f'./data/{dataset_name}_pretok_char.bin'
    # spm_model_path = f'./tokenizer/tok_{dataset_name}_char.model'
    # sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    # tok_list = np.memmap(tok_file, dtype=np.uint16, mode="r")
    # h = []
    # for T in range(2, 17):
    #     h1 = fixedTentropy(tok_list, T, tokenizer=sp)
    #     h.append(h1)
    #     print(f"----- fixed T = {T} | entropy = {h1} ------")
    # print(range(2, 17))
    # print(h)
    # h1 = fixedTentropy(tok_list, 16, tokenizer=sp)







