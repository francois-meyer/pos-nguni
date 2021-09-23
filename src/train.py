import re
import sys
import math
import string
import torch
from torch import nn, optim
import time
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import itertools
import json
import collections
import random
from sklearn.metrics import f1_score
import numpy as np
import nltk
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models import BiLSTM_CRF_Tagger, LSTMTagger, MorphLSTMTagger, Subword_BiLSTM_CRF_Tagger

START_TAG = "<START>"
STOP_TAG = "<STOP>"



def compute_scores(gold_tags, pred_tags, tag2index):
    total = 0
    correct = 0

    gold_tags_flat = []
    pred_tags_flat = []

    for i in range(len(pred_tags)):
        for j in range(len(pred_tags[i])):
            # this_word = index2word[sentence[i]]
            # if all(ch in string.punctuation for ch in this_word) or this_word == "<s>":
            #     num_test_tokens -= 1
            #     continue
            if gold_tags[i][j] == tag2index["<pad>"]:
                break

            total += 1
            if pred_tags[i][j] == gold_tags[i][j]:
                correct += 1

        seq_len = gold_tags[i].index(tag2index["<pad>"]) if tag2index["<pad>"] in gold_tags[i] else len(gold_tags[i])

        if len(gold_tags[i][0: seq_len]) != len(pred_tags[i][0: seq_len]):
            print(len(gold_tags[i][0: seq_len]), len(pred_tags[i][0: seq_len]))
            print(gold_tags[i])
            print(pred_tags[i])
            print("--------------------------------------------------------------")

        gold_tags_flat.extend(gold_tags[i][0: seq_len])
        pred_tags_flat.extend(pred_tags[i][0: seq_len])

    #gold_tags_flat = [item for ls in gold_tags for item in ls]
    #pred_tags_flat = [item for ls in pred_tags for item in ls]

    acc = (correct + 0.0) / total
    f1 = f1_score(y_true=gold_tags_flat, y_pred=pred_tags_flat, average="macro")

    return acc, f1


def grid_search():
    grid = {}
    grid["crf"] = [False]
    grid["subword"] = [True]
    grid["input_size"] = [512]
    grid["hidden_size"] = [512]
    grid["num_layers"] = [1, 2]
    grid["dropout"] = [0.2]
    grid["num_epochs"] = [30]
    grid["weight_decay"] = [1e-5]
    grid["lr_patience"] = [3]
    grid["lr"] = [0.01, 0.005, 0.001]
    grid["batch_size"] = [64]
    grid["clip"] = [1.0]
    grid["log_interval"] = [10]


    keys, values = zip(*grid.items())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_path = "log.txt"
    output_file = open(output_path, "w")
    output_file.close()

    grid_search = {}
    for i, params in enumerate(grid):
        print("%d/%d" % (i+1, len(grid)))
        start = time.time()

        acc, f1, epoch = cv(words, subwords, tags, params, folds=10)
        grid_search[json.dumps(params)] = {"f1": f1, "acc": acc, "epoch": epoch}

        end = time.time()
        print("CV completed in %fs.\n" % (end - start))

    sorted_grid = collections.OrderedDict(sorted(grid_search.items(), key=lambda t:t[1]["f1"], reverse=True))

    for key in sorted_grid:
        print(sorted_grid[key], "   ", key)


def tokenize_char_ngrams(word):
    # Split into all possible segments
    segs = []
    max_n = 2
    for n in range(max_n, max_n+1):
        chars = list(word)
        segs_n = nltk.ngrams(chars, n=n)
        segs_n = ["".join(seg) for seg in segs_n]
        #segs_n = [seg for seg in segs_n if seg.isalpha() and len(seg) == n]
        segs.extend(segs_n)
    return segs

def tokenize_chars(word):
    return list(word)


def read_raw_datasets(dataset_paths, tokenize):
    # Create lists of lists to read test corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    tags = []
    subwords = []

    for dataset_path in dataset_paths:
        lang_words = []
        lang_subwords = []
        lang_tags = []

        dataset = open(dataset_path, "r").read().split("\n")
        new_sent = True
        for line in dataset[1:]:  # discard first line, which is just column headings

            if line.strip() == "":
                continue

            word, morph, tag = line.rsplit("\t", maxsplit=2)
            if word == ".":
                new_sent = True
            elif new_sent:
                lang_words.append([])
                lang_subwords.append([])
                lang_tags.append([])
                new_sent = False

            lang_words[-1].append(word)
            lang_subwords[-1].append(tokenize(word))
            lang_tags[-1].append(tag)
        words.append(lang_words)
        subwords.append(lang_subwords)
        tags.append(lang_tags)

    return words, subwords, tags


def train2ids(train_words, train_subwords, train_tags):
    word2index = {}
    word2index["<pad>"] = 0
    word2index["<unk>"] = 1
    # word2index["<s>"] = 2

    subword2index = {}
    subword2index["<pad>"] = 0
    subword2index["<unk>"] = 1
    # word2index["<s>"] = 2

    tag2index = {}
    tag2index["<pad>"] = 0
    tag2index["<unk>"] = 1
    tag2index[START_TAG] = 2
    tag2index[STOP_TAG] = 3

    # tag2index["<s>"] = 2
    index2tag = ["<unk", "<pad>", START_TAG, STOP_TAG]  # "<s>"

    # Count words
    # Go throught training corpus line by line, reading words and tags into lists and creating vocabs
    word_counts = {}
    for i, sentence in enumerate(train_words):  # discard first line, which is just column headings
        for j, word in enumerate(sentence):
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 0

    # Create lists of lists to read corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    subwords = []
    tags = []

    # Go throught training corpus line by line, reading words and tags into lists and creating vocabs
    for i, sentence in enumerate(train_words):  # discard first line, which is just column headings

        words.append([])
        subwords.append([])
        tags.append([])

        for j, word in enumerate(sentence):

            # if word_counts[word] > 2:
            #     if not word in word2index:
            #         word2index[word] = len(word2index)
            #     words[-1].append(word2index[word])
            # else:
            #     words[-1].append(word2index["<unk>"])

            if not word in word2index:
                word2index[word] = len(word2index)
            words[-1].append(word2index[word])

            subwords[i].append([])
            for subword in train_subwords[i][j]:
                if not subword in subword2index:
                    subword2index[subword] = len(subword2index)
                subwords[i][j].append(subword2index[subword])

            tag = train_tags[i][j]
            if not tag in tag2index:
                tag2index[tag] = len(tag2index)
                index2tag.append(tag)
            tags[-1].append(tag2index[tag])

    return words, subwords, tags, word2index, subword2index, tag2index, index2tag


def dev2ids(dev_words, dev_subwords, dev_tags, word2index, subword2index, tag2index):
    # Create lists of lists to read corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    subwords = []
    tags = []

    # Go throught training corpus line by line, reading words and tags into lists and creating vocabs
    for i, sentence in enumerate(dev_words):  # discard first line, which is just column headings

        words.append([])
        subwords.append([])
        tags.append([])

        for j, word in enumerate(sentence):
            if not word in word2index:
                word = "<unk>"
            words[-1].append(word2index[word])

            subwords[i].append([])
            for subword in dev_subwords[i][j]:
                if not subword in subword2index:
                    subword2index[subword] = len(subword2index)
                subwords[i][j].append(subword2index[subword])

            tag = dev_tags[i][j]
            if not tag in tag2index:
                tag = "<unk>"
            tags[-1].append(tag2index[tag])

    return words, subwords, tags


def cv(words, subwords, tags, params, folds=10, track=False):

    output_path = "log.txt"
    output_file = open(output_path, "a")

    fold_sizes = []
    words_shuffled = []
    subwords_shuffled = []
    tags_shuffled = []

    for i in range(len(words)):
        lang_pairs = list(zip(words[i], subwords[i], tags[i]))
        random.shuffle(lang_pairs)

        lang_words_shuffled, lang_subwords_shuffled, lang_tags_shuffled = zip(*lang_pairs)
        words_shuffled.append(lang_words_shuffled)
        subwords_shuffled.append(lang_subwords_shuffled)
        tags_shuffled.append(lang_tags_shuffled)

        full_size = len(words[i])
        test_size = int(full_size * (1 / folds))
        train_size = full_size - test_size
        fold_sizes.append((train_size, test_size))

    accs = []
    f1s = []
    epochs = []
    output_file.write("\n\n------------------------------------------------------------\n")
    output_file.write("Model: %s\n" % (params))
    output_file.write("------------------------------------------------------------\n")
    for k in range(folds):

        if k + 1 > 2:
            break
        output_file.write("\n\n------------------------------------------------------------\n")
        output_file.write("Fold %d\n" % (k+1))
        output_file.write("------------------------------------------------------------\n")

        train_words = []
        dev_words = []
        train_subwords = []
        dev_subwords = []
        train_tags = []
        dev_tags = []

        for i in range(len(words)):
            dev_indices = list(range(k * fold_sizes[i][1], k * fold_sizes[i][1] + fold_sizes[i][1]))
            train_indices = [index for index in range(0, full_size) if index not in dev_indices]

            dev_words.extend([words_shuffled[i][idx] for idx in dev_indices])
            train_words.extend([words_shuffled[i][idx] for idx in train_indices])

            dev_subwords.extend([subwords_shuffled[i][idx] for idx in dev_indices])
            train_subwords.extend([subwords_shuffled[i][idx] for idx in train_indices])

            dev_tags.extend([tags_shuffled[i][idx] for idx in dev_indices])
            train_tags.extend([tags_shuffled[i][idx] for idx in train_indices])

        train_word_ids, train_subword_ids, train_tag_ids, word2index, subword2index, tag2index, index2tag = train2ids(train_words, train_subwords, train_tags)
        dev_word_ids, dev_subword_ids, dev_tag_ids = dev2ids(dev_words, dev_subwords, dev_tags, word2index, subword2index, tag2index)

        known_dev_word_ids = []
        known_dev_tag_ids = []
        unks = 0
        total = 0
        for i, sentence in enumerate(dev_word_ids):
            keep = True
            for word in sentence:
                total += 1
                if word == 1:
                    keep = False
                    unks += 1
            if keep:
                known_dev_word_ids.append(sentence)
                known_dev_tag_ids.append(dev_tag_ids[i])

        # print(unks, total)
        # print(len(dev_word_ids), len(known_dev_word_ids))

        acc, f1, epoch = train_model(train_word_ids, train_subword_ids, train_tag_ids, word2index, subword2index, tag2index,
                                     index2tag, dev_word_ids, dev_subword_ids, dev_tag_ids, params, output_file, track=True)
        accs.append(acc)
        f1s.append(f1)
        epochs.append(epoch)

    ave_acc = np.mean(acc)
    ave_f1 = np.mean(f1)
    ave_epoch = np.mean(epochs)

    output_file.close()
    return ave_acc, ave_f1, ave_epoch


def train_model(train_words, train_subwords, train_tags, word2index, subword2index, tag2index, index2tag, dev_words, dev_subwords, dev_tags, params, output_file, track=False):
    crf = params["crf"]
    subword = params["subword"]
    input_size = params["input_size"]
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    dropout = params["dropout"]
    num_epochs = params["num_epochs"]
    weight_decay = params["weight_decay"]
    lr_patience = params["lr_patience"]
    lr = params["lr"]
    batch_size = params["batch_size"]
    clip = params["clip"]
    log_interval = params["log_interval"]

    index2word = {v: k for k, v in word2index.items()}
    index2subword = {v: k for k, v in subword2index.items()}
    word_vocab_size = len(word2index)
    subword_vocab_size = len(subword2index)
    tag_vocab_size = len(tag2index)
    word_pad_id = word2index["<pad>"]
    subword_pad_id = subword2index["<pad>"]
    tag_pad_id = tag2index["<pad>"]
    num_tags = len(tag2index)

    # Set up model and training
    if crf and not subword:
        model = BiLSTM_CRF_Tagger(vocab_size=word_vocab_size, tagset_size=num_tags, embedding_dim=input_size,
                                  hidden_dim=hidden_size, num_rnn_layers=num_layers)

    elif crf and subword:
        model = Subword_BiLSTM_CRF_Tagger(word_vocab_size=word_vocab_size, subword_vocab_size=subword_vocab_size, tagset_size=num_tags, embedding_dim=input_size,
                                  hidden_dim=hidden_size, num_rnn_layers=num_layers)

    elif subword:
        model = MorphLSTMTagger(word_vocab_size, subword_vocab_size, tag_vocab_size, input_size, hidden_size, num_layers,
                                dropout, word_pad_id, subword_pad_id, bidirectional=True)
        criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_id)

    else:
        model = LSTMTagger(word_vocab_size, tag_vocab_size, input_size, hidden_size, num_layers, dropout, word_pad_id, bidirectional=True)
        criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_id)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # Adam with proper weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=lr_patience, verbose=True,
                                                     factor=0.5)  # Half the learning rate


    # Train model
    start = time.time()
    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        total_loss = 0

        if track:
            bar = tqdm(total=len(list(range(0, len(train_words), batch_size))), position=0, leave=True)
        for batch_num, batch_pos in enumerate(range(0, len(train_words), batch_size)):
            if track:
                bar.update(1)

            batch_words = train_words[batch_pos: batch_pos + batch_size]
            batch_words = [torch.tensor(batch_words[i]) for i in range(len(batch_words))]
            batch_words = pad_sequence(batch_words, padding_value=word_pad_id)

            batch_len = batch_words.shape[0]
            batch_subwords = train_subwords[batch_pos: batch_pos + batch_size]

            batch_subwords = [ls + [[subword_pad_id]]*(batch_len - len(ls)) for ls in batch_subwords]
            batch_subwords = [item for ls in batch_subwords for item in ls]
            batch_subwords = [torch.tensor(batch_subwords[i]) for i in range(len(batch_subwords))]
            batch_subwords = pad_sequence(batch_subwords, padding_value=word_pad_id)

            batch_tags = train_tags[batch_pos: batch_pos + batch_size]
            batch_tags = [torch.tensor(batch_tags[i]) for i in range(len(batch_tags))]
            batch_tags = pad_sequence(batch_tags, padding_value=tag_pad_id)

            model.train()
            model.zero_grad()

            if crf:
                loss = model.loss(batch_words.T, batch_subwords.T, batch_tags.T)
            else:
                logits = model(batch_words, batch_subwords)
                loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item()

            if batch_num % log_interval == 0 and batch_num > 0:  # and batch_num > 0:
                cur_loss = total_loss / log_interval
                if track:
                    print("| epoch {:3d} | loss {:5.2f} |".format(epoch, cur_loss))
                else:
                    output_file.write("| epoch {:3d} | loss {:5.2f} | \n".format(epoch, cur_loss))
                    output_file.flush()
                total_loss = 0

        ##############################################################################
        # Evaluate on development set
        model.eval()
        val_loss = 0.0
        gold_tags = []
        pred_tags = []

        with torch.no_grad():
            for batch_num, batch_pos in enumerate(range(0, len(dev_words), batch_size)):
                batch_words = dev_words[batch_pos: batch_pos + batch_size]
                batch_words = [torch.tensor(batch_words[i]) for i in range(len(batch_words))]
                batch_words = pad_sequence(batch_words, padding_value=word_pad_id)

                batch_len = batch_words.shape[0]
                batch_subwords = dev_subwords[batch_pos: batch_pos + batch_size]
                batch_subwords = [ls + [[subword_pad_id]] * (batch_len - len(ls)) for ls in batch_subwords]
                batch_subwords = [item for ls in batch_subwords for item in ls]
                batch_subwords = [torch.tensor(batch_subwords[i]) for i in range(len(batch_subwords))]
                batch_subwords = pad_sequence(batch_subwords, padding_value=word_pad_id)

                batch_tags = dev_tags[batch_pos: batch_pos + batch_size]
                batch_tags = [torch.tensor(batch_tags[i]) for i in range(len(batch_tags))]
                batch_tags = pad_sequence(batch_tags, padding_value=tag_pad_id)
                gold_tags.extend(batch_tags.T.tolist())

                if crf:
                    logits, tag_seq = model(batch_words.T, batch_subwords.T)
                    pred_tags.extend(tag_seq)
                else:
                    logits = model(batch_words, batch_subwords)
                    batch_pred_tags = torch.max(logits, dim=-1).indices.T
                    pred_tags.extend(batch_pred_tags.tolist())

                    loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
                    val_loss += loss.item()


        acc, f1 = compute_scores(gold_tags, pred_tags, tag2index)
        if track:
            print("| epoch {:3d} | valid loss {:5.2f} | acc {:5.4f} | f1 {:5.4f} |".format(epoch, val_loss, acc, f1))
        else:
            output_file.write("| epoch {:3d} | valid loss {:5.2f} | acc {:5.4f} | f1 {:5.4f} |\n".format(epoch, val_loss, acc, f1))
        # generate_text(model, vocab, "Tears will gather ", gen_len=100)
        scheduler.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_epoch = epoch

    end = time.time()
    output_file.write("Training completed in %fs.\n" % (end - start))
    output_file.flush()

    if track:
        print("BEST RESULT:")
        print("| epoch {:3d} | acc {:5.4f} | f1 {:5.4f}".format(best_epoch, best_acc, best_f1))

    return best_acc, best_f1, best_epoch

if __name__ == '__main__':
    params = {}
    params["crf"] = True
    params["subword"] = True
    params["input_size"] = 128
    params["hidden_size"] = 512
    params["num_layers"] = 1
    params["dropout"] = 0.2
    params["num_epochs"] = 50
    params["weight_decay"] = 1e-5
    params["lr_patience"] = 3
    params["lr"] = 0.01
    params["batch_size"] = 64
    params["clip"] = 1.0
    params["log_interval"] = 10
    #train_one_model(params)

    train_paths = ["../data/train/xh.gold.train"]#, "../data/train/zu.gold.train", "../data/train/nr.gold.train", "../data/train/ss.gold.train"]
    # dev_paths = ["../data/dev/xh.gold.dev", "../data/dev/zu.gold.dev", "../data/dev/nr.gold.dev", "../data/dev/ss.gold.dev"]

    #train_words, train_tags, word2index, tag2index, index2tag = read_train_datasets(train_paths)
    #dev_words, dev_tags = read_dev_datasets(dev_paths, word2index, tag2index)

    output_path = "log.txt"
    output_file = open(output_path, "w")
    output_file.close()

    words, subwords, tags = read_raw_datasets(dataset_paths=train_paths, tokenize=tokenize_chars)
    cv(words, subwords, tags, params, folds=10, track=False)
    #grid_search()

