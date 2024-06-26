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
import random

from models import MorphLSTMTagger, Subword_BiLSTM_CRF_Tagger

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


def grid_search(skip=0):
    grid = {}
    grid["crf"] = [True, False]
    grid["comp"] = ["sum"]
    grid["input_size"] = [512]
    grid["hidden_size"] = [512]
    grid["num_layers"] = [1, 2]
    grid["dropout"] = [0.2]
    grid["num_epochs"] = [1]
    grid["weight_decay"] = [1e-5]
    grid["lr_patience"] = [3]
    grid["lr"] = [0.01]
    grid["batch_size"] = [32]
    grid["clip"] = [1.0]
    grid["log_interval"] = [10]

    # grid["crf"] = [True, False]
    # grid["comp"] = ["lstm", "sum"]
    # grid["input_size"] = [512]
    # grid["hidden_size"] = [1024, 512]
    # grid["num_layers"] = [1, 2]
    # grid["dropout"] = [0.2, 0.5]
    # grid["num_epochs"] = [30]
    # grid["weight_decay"] = [1e-5]
    # grid["lr_patience"] = [3]
    # grid["lr"] = [0.001]
    # grid["batch_size"] = [64]
    # grid["clip"] = [1.0]
    # grid["log_interval"] = [10]

    keys, values = zip(*grid.items())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_path = "log.txt"
    output_file = open(output_path, "a")
    #output_file.close()

    grid_search = {}
    for i, params in enumerate(grid):
        if i + 1 <= skip:
            continue
        print("%d/%d" % (i+1, len(grid)))
        start = time.time()

        acc, f1, epoch = cv(words, subwords, tags, params, folds=10)
        grid_search[json.dumps(params)] = {"f1": f1, "acc": acc, "epoch": epoch}
        print(json.dumps(grid_search[json.dumps(params)]) + " , " +  json.dumps(params))

        end = time.time()
        print("CV completed in %fs.\n" % (end - start))

    sorted_grid = collections.OrderedDict(sorted(grid_search.items(), key=lambda t:t[1]["f1"], reverse=True))

    for key in sorted_grid:
        print(sorted_grid[key], "   ", key)


def tokenize_char_ngrams(word):
    # Split into all possible segment
    if not word.isalpha():
        return list(word)
    if MAX_N == 2:
        word = "<" + word + ">"
    elif MAX_N == 3:
        word = "<<" + word + ">>"
    elif MAX_N == 4:
        word = "<<<" + word + ">>>"
    segs = []
    chars = list(word)
    segs_n = nltk.ngrams(chars, n=MAX_N)
    segs_n = ["".join(seg) for seg in segs_n]
    #segs_n = [seg for seg in segs_n if seg.isalpha() and len(seg) == n]
    segs.extend(segs_n)
    return segs

def tokenize_chars(word):
    return list(word)


def read_raw_train_dataset(dataset_path, tokenize):
    # Create lists of lists to read test corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    tags = []
    subwords = []

    dataset = open(dataset_path, "r").read().split("\n")
    new_sent = True
    for line in dataset[1:]:  # discard first line, which is just column headings

        if line.strip() == "":
            continue

        word, morph, tag = line.rsplit("\t", maxsplit=2)
        if word == ".":
            new_sent = True
        elif new_sent:
            words.append([])
            subwords.append([])
            tags.append([])
            new_sent = False

        words[-1].append(word)
        subwords[-1].append(tokenize(word))
        tags[-1].append(tag)

    return words, subwords, tags


def read_raw_test_dataset(dataset_path, tokenize):
    # Create lists of lists to read test corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    subwords = []

    dataset = open(dataset_path, "r").read().split("\n")
    new_sent = True
    for line in dataset[1:]:  # discard first line, which is just column headings

        if line.strip() == "":
            continue

        word, morphs, lemma, xpos, tag = line.rsplit("\t", maxsplit=4)
        if word == ".":
            new_sent = True
        elif new_sent:
            words.append([])
            subwords.append([])
            new_sent = False

        words[-1].append(word)
        subwords[-1].append(tokenize(word))

    return words, subwords



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
                    subword = "<unk>"
                subwords[i][j].append(subword2index[subword])

            tag = dev_tags[i][j]
            if not tag in tag2index:
                tag = "<unk>"
            tags[-1].append(tag2index[tag])

    return words, subwords, tags


def test2ids(test_words, test_subwords, word2index, subword2index):
    # Create lists of lists to read corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    subwords = []

    # Go throught training corpus line by line, reading words and tags into lists and creating vocabs
    for i, sentence in enumerate(test_words):  # discard first line, which is just column headings

        words.append([])
        subwords.append([])

        for j, word in enumerate(sentence):
            if not word in word2index:
                word = "<unk>"
            words[-1].append(word2index[word])

            subwords[i].append([])
            for subword in test_subwords[i][j]:
                if not subword in subword2index:
                    subword = "<unk>"
                subwords[i][j].append(subword2index[subword])

    return words, subwords


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

        if k + 1 > 3:
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

    #output_file.close()
    return ave_acc, ave_f1, ave_epoch


def train_model(train_words, train_subwords, train_tags, word2index, subword2index, tag2index, index2tag, dev_words, dev_subwords, dev_tags, params, output_file, track=False, mode="tune"):
    crf = params["crf"]
    comp = params["comp"]
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
    if crf:
        model = Subword_BiLSTM_CRF_Tagger(word_vocab_size=word_vocab_size, subword_vocab_size=subword_vocab_size, tagset_size=num_tags, embedding_dim=input_size,
                                  hidden_dim=hidden_size, dropout=dropout, num_rnn_layers=num_layers, comp=comp)

    else:
        model = MorphLSTMTagger(word_vocab_size, subword_vocab_size, tag_vocab_size, input_size, hidden_size, num_layers,
                                dropout, word_pad_id, subword_pad_id, bidirectional=True, comp=comp)
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

        print(epoch)
        total_loss = 0

        epoch_train_words = []
        epoch_train_subwords = []
        epoch_train_tags = []

        for i, sentence in enumerate(train_words):
            epoch_train_words.append([])
            epoch_train_tags.append([])
            for j, word in enumerate(sentence):
                if random.uniform(0, 1) > 0.1:
                    epoch_train_words[-1].append(word)
                else:
                    epoch_train_words[-1].append(word2index["<unk>"])
                epoch_train_tags[-1].append(train_tags[i][j])

        for sentence in train_subwords:
            epoch_train_subwords.append([])
            for word in sentence:
                epoch_train_subwords[-1].append([])
                for subword in word:
                    if random.uniform(0, 1) > 0.05: #83
                        epoch_train_subwords[-1][-1].append(subword)
                    else:
                        epoch_train_subwords[-1][-1].append(subword2index["<unk>"])

        triplet = list(zip(epoch_train_words, epoch_train_subwords, epoch_train_tags))
        epoch_train_words, epoch_train_subwords, epoch_train_tags = zip(*triplet)


        if track:
            bar = tqdm(total=len(list(range(0, len(train_words), batch_size))), position=0, leave=True)
        for batch_num, batch_pos in enumerate(range(0, len(train_words), batch_size)):
            if track:
                bar.update(1)

            batch_words = epoch_train_words[batch_pos: batch_pos + batch_size]
            batch_words = [torch.tensor(batch_words[i]) for i in range(len(batch_words))]
            batch_words = pad_sequence(batch_words, padding_value=word_pad_id)

            batch_len = batch_words.shape[0]
            batch_subwords = epoch_train_subwords[batch_pos: batch_pos + batch_size]

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
                loss = model.loss(batch_words.T, batch_subwords, batch_tags.T)
            else:
                logits = model(batch_words, batch_subwords)
                loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item()

            if batch_num % log_interval == 0 and batch_num > 0:  # and batch_num > 0:
                cur_loss = total_loss / log_interval
                # TODO: uncomment
                # print("| epoch {:3d} | loss {:5.2f} |".format(epoch, cur_loss))
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
                    logits, tag_seq = model(batch_words.T, batch_subwords)
                    pred_tags.extend(tag_seq)
                else:
                    logits = model(batch_words, batch_subwords)
                    batch_pred_tags = torch.max(logits, dim=-1).indices.T
                    pred_tags.extend(batch_pred_tags.tolist())

                    loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
                    val_loss += loss.item()


        acc, f1 = compute_scores(gold_tags, pred_tags, tag2index)
        # TODO: uncomment
        #print("| epoch {:3d} | valid loss {:5.2f} | acc {:5.4f} | f1 {:5.4f} |".format(epoch, val_loss, acc, f1))




        # generate_text(model, vocab, "Tears will gather ", gen_len=100)
        #scheduler.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_epoch = epoch

        if epoch + 1 == 6 or epoch + 1 == 9 or epoch + 1 == 12 or epoch + 1 == 15:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5


    end = time.time()
    #print("Training completed in %fs.\n" % (end - start))


    if track:
        print("BEST RESULT:")
        print("| epoch {:3d} | acc {:5.4f} | f1 {:5.4f}".format(best_epoch, best_acc, best_f1))

    if mode == "test":
        return model, "{:.2f}".format(acc * 100), "{:.2f}".format(f1 * 100)

    return best_acc, best_f1, best_epoch


def test_model(train_words, train_subwords, train_tags, test_words, test_subwords, predict_path):
    crf = params["crf"]
    batch_size = params["batch_size"]

    predict_file = open(predict_path, "w")
    predict_file.close()
    predict_file = open(predict_path, "a")

    accs = []
    f1s = []
    epochs = []

    train_word_ids, train_subword_ids, train_tag_ids, word2index, subword2index, tag2index, index2tag = train2ids(train_words, train_subwords, train_tags)
    test_word_ids, test_subword_ids = test2ids(test_words, test_subwords, word2index, subword2index)
    word_pad_id = word2index["<pad>"]
    subword_pad_id = subword2index["<pad>"]


    model, acc, f1 = train_model(train_word_ids, train_subword_ids, train_tag_ids, word2index, subword2index, tag2index,
                        index2tag, train_word_ids, train_subword_ids, train_tag_ids, params, output_file, track=False, mode="test")

    return acc, f1

    # predict_file.write("TOKEN\tUPOS\n")
    #
    # pred_tags = []
    #
    # with torch.no_grad():
    #     for batch_num, batch_pos in enumerate(range(0, len(test_word_ids), batch_size)):
    #         batch_words = test_word_ids[batch_pos: batch_pos + batch_size]
    #         batch_words = [torch.tensor(batch_words[i]) for i in range(len(batch_words))]
    #         batch_words = pad_sequence(batch_words, padding_value=word_pad_id)
    #
    #         batch_len = batch_words.shape[0]
    #         batch_subwords = test_subword_ids[batch_pos: batch_pos + batch_size]
    #         batch_subwords = [ls + [[subword_pad_id]] * (batch_len - len(ls)) for ls in batch_subwords]
    #         batch_subwords = [item for ls in batch_subwords for item in ls]
    #         batch_subwords = [torch.tensor(batch_subwords[i]) for i in range(len(batch_subwords))]
    #         batch_subwords = pad_sequence(batch_subwords, padding_value=subword_pad_id)
    #
    #         if crf:
    #             logits, tag_seq = model(batch_words.T, batch_subwords)
    #             pred_tags.extend(tag_seq)
    #         else:
    #             logits = model(batch_words, batch_subwords)
    #             batch_pred_tags = torch.max(logits, dim=-1).indices.T
    #             pred_tags.extend(batch_pred_tags.tolist())
    #
    #
    # for i, sentence in enumerate(pred_tags):
    #     for j, tag in enumerate(sentence):
    #         predict_file.write("%s\t%s\n" % (test_words[i][j], index2tag[tag]))


if __name__ == '__main__':
    params = {}
    params["crf"] = True
    params["comp"] = "sum"
    params["input_size"] = 512
    params["hidden_size"] = 512
    params["num_layers"] = 1
    params["dropout"] = 0.2
    params["num_epochs"] = 15
    params["weight_decay"] = 1e-5
    params["lr_patience"] = 3
    params["lr"] = 0.01
    params["batch_size"] = 64
    params["clip"] = 1.0
    params["log_interval"] = 10
    #train_one_model(params)  ##87

    output_path = "log.txt"
    output_file = open(output_path, "w")
    output_file.close()
    predict_path = "prediction.tsv"
    all_train_paths = ["../data/train/nr.gold.train", "../data/train/xh.gold.train", "../data/train/zu.gold.train", "../data/train/ss.gold.train"]
    all_test_paths = ["../data/gold/nr.gold.test", "../data/gold/xh.gold.test", "../data/gold/zu.gold.test",  "../data/gold/ss.gold.test"]

    system_results1 = ""
    system_results2 = ""
    system_results3 = ""
    system_results4 = ""
    for j in range(len(all_train_paths)):
        print(all_train_paths[j])

        # char + sum
        params["comp"] = "sum"
        test_words, test_subwords = read_raw_test_dataset(all_test_paths[0], tokenize=tokenize_chars)
        train_words, train_subwords, train_tags = read_raw_train_dataset(dataset_path=all_train_paths[0],  tokenize=tokenize_chars)
        acc, f1 = test_model(train_words, train_subwords, train_tags, test_words, test_subwords, predict_path)
        system_results1 += " & " + acc + " & " + f1

        # char + lstm
        params["comp"] = "lstm"
        predict_path = "prediction.tsv"
        test_words, test_subwords = read_raw_test_dataset(all_test_paths[0], tokenize=tokenize_chars)
        train_words, train_subwords, train_tags = read_raw_train_dataset(dataset_path=all_train_paths[0], tokenize=tokenize_chars)
        acc, f1 = test_model(train_words, train_subwords, train_tags, test_words, test_subwords, predict_path)
        system_results2 += " & " + acc + " & " + f1

        # ngram + sum
        MAX_N = 2
        params["comp"] = "sum"
        predict_path = "prediction.tsv"
        test_words, test_subwords = read_raw_test_dataset(all_test_paths[0], tokenize=tokenize_char_ngrams)
        train_words, train_subwords, train_tags = read_raw_train_dataset(dataset_path=all_train_paths[0],  tokenize=tokenize_char_ngrams)
        acc, f1 = test_model(train_words, train_subwords, train_tags, test_words, test_subwords, predict_path)
        system_results3 += " & " + acc + " & " + f1

        # ngram + lstm
        MAX_N = 2
        params["comp"] = "lstm"
        predict_path = "prediction.tsv"
        test_words, test_subwords = read_raw_test_dataset(all_test_paths[0], tokenize=tokenize_char_ngrams)
        train_words, train_subwords, train_tags = read_raw_train_dataset(dataset_path=all_train_paths[0], tokenize=tokenize_char_ngrams)
        acc, f1 = test_model(train_words, train_subwords, train_tags, test_words, test_subwords, predict_path)
        system_results4 += " & " + acc + " & " + f1



    print(system_results1)
    print(system_results2)
    print(system_results3)
    print(system_results4)



    train_paths = ["../data/train/xh.gold.train"]#, "../data/train/zu.gold.train", "../data/train/nr.gold.train", "../data/train/ss.gold.train"]
    # dev_paths = ["../data/dev/xh.gold.dev", "../data/dev/zu.gold.dev", "../data/dev/nr.gold.dev", "../data/dev/ss.gold.dev"]
    test_paths = ["../data/gold/xh.gold.test"] #, "../data/test/zu.test", "../data/test/nr.test", "../data/test/ss.test"]




    #train_words, train_tags, word2index, tag2index, index2tag = read_train_datasets(train_paths)
    #dev_words, dev_tags = read_dev_datasets(dev_paths, word2index, tag2index)



    # MAX_N = 3
    # train_words, train_subwords, train_tags = read_raw_datasets(dataset_paths=train_paths, tokenize=tokenize_char_ngrams)
    # test_path =
    # test_model(train_words, train_subwords, train_tags, params, test_path, predict_path)

    # print("CHARACTERS")
    # words, subwords, tags = read_raw_datasets(dataset_paths=train_paths, tokenize=tokenize_chars)
    # #cv(words, subwords, tags, params, folds=10, track=False)
    # grid_search()




    #
    #

    # MAX_N = 2
    # train_words, train_subwords, train_tags = read_raw_datasets(dataset_paths=train_paths, tokenize=tokenize_char_ngrams)
    # #grid_search(skip=2)
    # cv(train_words, train_subwords, train_tags, params, folds=10, track=False)



    # print("NGRAMS=3")
    # MAX_N = 3
    # words, subwords, tags = read_raw_datasets(dataset_paths=train_paths, tokenize=tokenize_char_ngrams)
    # grid_search()
    #
    # print("NGRAMS=4")
    # MAX_N = 4
    # words, subwords, tags = read_raw_datasets(dataset_paths=train_paths, tokenize=tokenize_char_ngrams)
    # grid_search()


    # print("BEGIN SISWATI")
    #
    # train_paths = ["../data/train/ss.gold.train"]
    # print("CHARACTERS")
    # words, subwords, tags = read_raw_datasets(dataset_paths=train_paths, tokenize=tokenize_chars)
    # # cv(words, subwords, tags, params, folds=10, track=False)
    # grid_search()
    #
    # print("NGRAMS=2")
    # MAX_N = 2
    # words, subwords, tags = read_raw_datasets(dataset_paths=train_paths, tokenize=tokenize_char_ngrams)
    # grid_search()
    #
    # print("NGRAMS=3")
    # MAX_N = 3
    # words, subwords, tags = read_raw_datasets(dataset_paths=train_paths, tokenize=tokenize_char_ngrams)
    # grid_search()




