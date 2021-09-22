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


def read_train_dataset(dataset_path, words, tags, word2index, tag2index, index2tag):
    dataset = open(dataset_path, "r").read().split("\n")

    # Go throught training corpus line by line, reading words and tags into lists and creating vocabs
    new_sent = True
    for line in dataset[1:]:  # discard first line, which is just column headings

        if line.strip() == "":
            continue

        word, morphs, tag = line.rsplit("\t", maxsplit=2)
        if word == ".":
            new_sent = True
        elif new_sent:
            words.append([])
            tags.append([])
            new_sent = False

        if not word in word2index:
            word2index[word] = len(word2index)
        words[-1].append(word2index[word])
        if not tag in tag2index:
            tag2index[tag] = len(tag2index)
            index2tag.append(tag)
        tags[-1].append(tag2index[tag])


def read_train_datasets(dataset_paths):

    word2index = {}
    word2index["<unk>"] = 0
    word2index["<pad>"] = 1
    # word2index["<s>"] = 2

    tag2index = {}
    tag2index["<unk>"] = 0
    tag2index["<pad>"] = 1
    # tag2index["<s>"] = 2
    index2tag = ["<unk", "<pad>"]  # "<s>"

    # Create lists of lists to read corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    tags = []

    for dataset_path in dataset_paths:
        read_train_dataset(dataset_path, words, tags, word2index, tag2index, index2tag)
        print("Finished reading in %s." % dataset_path)

    return words, tags, word2index, tag2index, index2tag


def read_dev_dataset(dataset_path, words, tags, word2index, tag2index):
    dataset = open(dataset_path, "r").read().split("\n")
    new_sent = True
    for line in dataset[1:]:  # discard first line, which is just column headings

        if line.strip() == "":
            continue

        word, morphs, tag = line.rsplit("\t", maxsplit=2)
        if word == ".":
            new_sent = True
        elif new_sent:
            words.append([])
            tags.append([])
            new_sent = False

        if not word in word2index:
            word = "<unk>"
        words[-1].append(word2index[word])

        if not tag in tag2index:
            tag = "<unk>"
        tags[-1].append(tag2index[tag])


def read_dev_datasets(dataset_paths, word2index, tag2index):
    # Create lists of lists to read test corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    tags = []

    for dataset_path in dataset_paths:
        read_dev_dataset(dataset_path, words, tags, word2index, tag2index)
        print("Finished reading in %s." % dataset_path)

    return words, tags


class LSTMTagger(nn.Module):
    """
    Vanilla LSTM model for POS tagging
    """
    def __init__(self, word_vocab_size, tag_vocab_size, input_size, hidden_size, num_layers, dropout, input_pad_id, bidirectional):
        super(LSTMTagger, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.tag_vocab_size = tag_vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.embedding = nn.Embedding(word_vocab_size, input_size, padding_idx=input_pad_id)
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)

        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, tag_vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        embeddings = self.drop(embeddings)
        hidden_states, final_states = self.lstm(embeddings)
        output = self.drop(hidden_states)
        logits = self.fc(output)
        return logits

    def init_states(self, batch_size):
        if self.bidirectional:
            init_states = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                           torch.zeros(self.num_layers, batch_size, self.hidden_size),
                           torch.zeros(self.num_layers, batch_size, self.hidden_size),
                           torch.zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            init_states = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                           torch.zeros(self.num_layers, batch_size, self.hidden_size))

        return init_states




def compute_scores(gold_tags, pred_tags, tag2index):
    total = 0
    correct = 0
    gold_tags_flat = []
    pred_tags_flat = []
    for i in range(len(gold_tags)):
        for j in range(len(gold_tags[i])):
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

    acc = (correct + 0.0) / total
    f1 = f1_score(y_true=gold_tags_flat, y_pred=pred_tags_flat, average="macro")

    return acc, f1


def train_model(train_words, train_tags, word2index, tag2index, index2tag, dev_words, dev_tags, params, output_file, track=False):
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
    word_vocab_size = len(word2index)
    tag_vocab_size = len(tag2index)
    word_pad_id = word2index["<pad>"]
    tag_pad_id = tag2index["<pad>"]
    num_tags = len(tag2index)

    # Set up model and training
    model = LSTMTagger(word_vocab_size, tag_vocab_size, input_size, hidden_size, num_layers, dropout, word_pad_id, bidirectional=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # Adam with proper weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience, verbose=True,
                                                     factor=0.5)  # Half the learning rate
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_id)

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

            batch_words = train_words[batch_pos: batch_pos + batch_size]
            if track:
                bar.update(1)
            batch_words = [torch.tensor(batch_words[i]) for i in range(len(batch_words))]
            batch_words = pad_sequence(batch_words, padding_value=word_pad_id)

            batch_tags = train_tags[batch_pos: batch_pos + batch_size]
            batch_tags = [torch.tensor(batch_tags[i]) for i in range(len(batch_tags))]
            batch_tags = pad_sequence(batch_tags, padding_value=tag_pad_id)

            model.train()
            model.zero_grad()

            logits = model(batch_words)

            loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
            # nll = - torch.sum(log_alpha) / torch.numel(input_ids)
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

                batch_tags = dev_tags[batch_pos: batch_pos + batch_size]
                batch_tags = [torch.tensor(batch_tags[i]) for i in range(len(batch_tags))]
                batch_tags = pad_sequence(batch_tags, padding_value=tag_pad_id)
                gold_tags.extend(batch_tags.T.tolist())

                logits = model(batch_words)
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
        scheduler.step(val_loss)
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



def grid_search():
    grid = {}
    grid["input_size"] = [512]
    grid["hidden_size"] = [512, 1024]
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

        acc, f1, epoch = cv(words, tags, params, folds=10)
        grid_search[json.dumps(params)] = {"f1": f1, "acc": acc, "epoch": epoch}

        end = time.time()
        print("CV completed in %fs.\n" % (end - start))

    sorted_grid = collections.OrderedDict(sorted(grid_search.items(), key=lambda t:t[1]["f1"], reverse=True))

    for key in sorted_grid:
        print(sorted_grid[key], "   ", key)



def read_raw_datasets(dataset_paths):
    # Create lists of lists to read test corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    tags = []

    for dataset_path in dataset_paths:
        lang_words = []
        lang_tags = []

        dataset = open(dataset_path, "r").read().split("\n")
        new_sent = True
        for line in dataset[1:]:  # discard first line, which is just column headings

            if line.strip() == "":
                continue

            word, morphs, tag = line.rsplit("\t", maxsplit=2)
            if word == ".":
                new_sent = True
            elif new_sent:
                lang_words.append([])
                lang_tags.append([])
                new_sent = False

            lang_words[-1].append(word)
            lang_tags[-1].append(tag)
        words.append(lang_words)
        tags.append(lang_tags)

    return words, tags


def train2ids(train_words, train_tags):
    word2index = {}
    word2index["<unk>"] = 0
    word2index["<pad>"] = 1
    # word2index["<s>"] = 2

    tag2index = {}
    tag2index["<unk>"] = 0
    tag2index["<pad>"] = 1
    # tag2index["<s>"] = 2
    index2tag = ["<unk", "<pad>"]  # "<s>"

    # Create lists of lists to read corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    tags = []

    # Go throught training corpus line by line, reading words and tags into lists and creating vocabs
    for i, sentence in enumerate(train_words):  # discard first line, which is just column headings

        words.append([])
        tags.append([])

        for j, word in enumerate(sentence):
            if not word in word2index:
                word2index[word] = len(word2index)
            words[-1].append(word2index[word])

            tag = train_tags[i][j]
            if not tag in tag2index:
                tag2index[tag] = len(tag2index)
                index2tag.append(tag)
            tags[-1].append(tag2index[tag])

    return words, tags, word2index, tag2index, index2tag


def dev2ids(dev_words, dev_tags, word2index, tag2index):
    # Create lists of lists to read corpus into
    # [[word11, word12, ...], [word21, word22, ...]]
    # [[tag11, tag12, ...], [tag21, tag22, ...]]
    words = []
    tags = []

    # Go throught training corpus line by line, reading words and tags into lists and creating vocabs
    for i, sentence in enumerate(dev_words):  # discard first line, which is just column headings

        words.append([])
        tags.append([])

        for j, word in enumerate(sentence):
            if not word in word2index:
                word = "<unk>"
            words[-1].append(word2index[word])

            tag = dev_tags[i][j]
            if not tag in tag2index:
                tag = "<unk>"
            tags[-1].append(tag2index[tag])

    return words, tags


def cv(words, tags, params, folds=10, track=False):

    output_path = "log.txt"
    output_file = open(output_path, "a")

    fold_sizes = []
    words_shuffled = []
    tags_shuffled = []

    for i in range(len(words)):
        lang_pairs = list(zip(words[i], tags[i]))
        random.shuffle(lang_pairs)

        lang_words_shuffled, lang_tags_shuffled = zip(*lang_pairs)
        words_shuffled.append(lang_words_shuffled)
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

        if k + 1 > 1:
            break
        output_file.write("\n\n------------------------------------------------------------\n")
        output_file.write("Fold %d\n" % (k+1))
        output_file.write("------------------------------------------------------------\n")

        train_words = []
        dev_words = []
        train_tags = []
        dev_tags = []
        for i in range(len(words)):
            dev_indices = list(range(k * fold_sizes[i][1], k * fold_sizes[i][1] + fold_sizes[i][1]))
            train_indices = [index for index in range(0, full_size) if index not in dev_indices]

            dev_words.extend([words_shuffled[i][idx] for idx in dev_indices])
            train_words.extend([words_shuffled[i][idx] for idx in train_indices])

            dev_tags.extend([tags_shuffled[i][idx] for idx in dev_indices])
            train_tags.extend([tags_shuffled[i][idx] for idx in train_indices])

        train_word_ids, train_tag_ids, word2index, tag2index, index2tag = train2ids(train_words, train_tags)
        dev_word_ids, dev_tag_ids = dev2ids(dev_words, dev_tags, word2index, tag2index)

        acc, f1, epoch = train_model(train_word_ids, train_tag_ids, word2index, tag2index, index2tag, dev_word_ids, dev_tag_ids, params, output_file, track=True)
        accs.append(acc)
        f1s.append(f1)
        epochs.append(epoch)

    ave_acc = np.mean(acc)
    ave_f1 = np.mean(f1)
    ave_epoch = np.mean(epochs)

    output_file.close()
    return ave_acc, ave_f1, ave_epoch


if __name__ == '__main__':
    params = {}
    params["input_size"] = 512
    params["hidden_size"] = 512
    params["num_layers"] = 2
    params["dropout"] = 0.2
    params["num_epochs"] = 50
    params["weight_decay"] = 1e-5
    params["lr_patience"] = 2
    params["lr"] = 0.005
    params["batch_size"] = 64
    params["clip"] = 1.0
    params["log_interval"] = 10
    #train_one_model(params)

    train_paths = ["../data/train/xh.gold.train", "../data/train/zu.gold.train", "../data/train/nr.gold.train", "../data/train/ss.gold.train"]
    # dev_paths = ["../data/dev/xh.gold.dev", "../data/dev/zu.gold.dev", "../data/dev/nr.gold.dev", "../data/dev/ss.gold.dev"]

    #train_words, train_tags, word2index, tag2index, index2tag = read_train_datasets(train_paths)
    #dev_words, dev_tags = read_dev_datasets(dev_paths, word2index, tag2index)

    output_path = "log.txt"
    output_file = open(output_path, "w")
    output_file.close()

    words, tags = read_raw_datasets(dataset_paths=train_paths)
    cv(words, tags, params, folds=10, track=False)
    #grid_search()


