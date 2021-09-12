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
    def __init__(self, word_vocab_size, tag_vocab_size, input_size, hidden_size, num_layers, dropout, input_pad_id):
        super(LSTMTagger, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.tag_vocab_size = tag_vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.embedding = nn.Embedding(word_vocab_size, input_size, padding_idx=input_pad_id)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, tag_vocab_size)

    def forward(self, input_ids, init_state_h, init_state_c):
        embeddings = self.embedding(input_ids)
        embeddings = self.drop(embeddings)
        hidden_states, final_states = self.lstm(embeddings, (init_state_h, init_state_c))
        output = self.drop(hidden_states)
        logits = self.fc(output)
        return logits

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))



def compute_scores(gold_tags, pred_tags, tag2index):
    total = 0
    correct = 0

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

    acc = (correct + 0.0) / total
    return acc


def train_model(train_words, train_tags, word2index, tag2index, index2tag, dev_words, dev_tags, params):
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
    model = LSTMTagger(word_vocab_size, tag_vocab_size, input_size, hidden_size, num_layers, dropout, word_pad_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # Adam with proper weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience, verbose=True,
                                                     factor=0.5)  # Half the learning rate
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_id)

    # Train model
    start = time.time()
    best_loss = float("inf")
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        total_loss = 0

        bar = tqdm(total=len(list(range(0, len(train_words), batch_size))), position=0, leave=True)
        for batch_num, batch_pos in enumerate(range(0, len(train_words), batch_size)):

            batch_words = train_words[batch_pos: batch_pos + batch_size]
            bar.update(1)
            batch_words = [torch.tensor(batch_words[i]) for i in range(len(batch_words))]
            batch_words = pad_sequence(batch_words, padding_value=word_pad_id)

            batch_tags = train_tags[batch_pos: batch_pos + batch_size]
            batch_tags = [torch.tensor(batch_tags[i]) for i in range(len(batch_tags))]
            batch_tags = pad_sequence(batch_tags, padding_value=tag_pad_id)

            model.train()
            model.zero_grad()

            init_state_h, init_state_c = model.init_states(batch_words.shape[-1])
            logits = model(batch_words, init_state_h, init_state_c)

            loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
            # nll = - torch.sum(log_alpha) / torch.numel(input_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item()

            if batch_num % log_interval == 0 and batch_num > 0:  # and batch_num > 0:
                cur_loss = total_loss / log_interval
                print("| epoch {:3d} | loss {:5.2f} | ".format(epoch, cur_loss))
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

                init_state_h, init_state_c = model.init_states(batch_words.shape[-1])
                logits = model(batch_words, init_state_h, init_state_c)
                batch_pred_tags = torch.max(logits, dim=-1).indices.T
                pred_tags.extend(batch_pred_tags.tolist())

                loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
                val_loss += loss.item()

        acc = compute_scores(gold_tags, pred_tags, tag2index)
        print("| epoch {:3d} | valid loss {:5.2f} | acc {:5.2f} | ".format(epoch, val_loss, acc))
        # generate_text(model, vocab, "Tears will gather ", gen_len=100)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch

    end = time.time()
    print("Training completed in %fs." % (end - start))

    return best_loss, acc, best_epoch
    # print("| epoch {:3d} | valid loss {:5.2f} | ".format(best_epoch, best_loss))


def grid_search():
    grid = {}
    grid["input_size"] = [128, 256]
    grid["hidden_size"] = [256, 512]
    grid["num_layers"] = [1]
    grid["dropout"] = [0.2]
    grid["num_epochs"] = [1]
    grid["weight_decay"] = [1e-5]
    grid["lr_patience"] = [2]
    grid["lr"] = [0.01]
    grid["batch_size"] = [64]
    grid["clip"] = [1.0]
    grid["log_interval"] = [10]

    keys, values = zip(*grid.items())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    grid_search = {}
    for params in grid:
        loss, acc, epoch = train_one_model(params)
        grid_search[json.dumps(params)] = {"loss": loss, "acc": acc, "epoch": epoch}

    sorted_grid = collections.OrderedDict(sorted(grid_search.items(), key=lambda t:t[1]["acc"], reverse=True))

    for key in sorted_grid:
        print(sorted_grid[key])


def train_one_model(params):
    return train_model(train_words, train_tags, word2index, tag2index, index2tag, dev_words, dev_tags, params)


if __name__ == '__main__':
    params = {}
    params["input_size"] = 128
    params["hidden_size"] = 256
    params["num_layers"] = 1
    params["dropout"] = 0.2
    params["num_epochs"] = 50
    params["weight_decay"] = 1e-5
    params["lr_patience"] = 2
    params["lr"] = 0.01
    params["batch_size"] = 64
    params["clip"] = 1.0
    params["log_interval"] = 10
    #train_one_model(params)

    train_paths = ["../data/train/xh.gold.train", "../data/train/zu.gold.train", "../data/train/nr.gold.train",
                   "../data/train/ss.gold.train"]
    dev_paths = ["../data/dev/xh.gold.dev", "../data/dev/zu.gold.dev", "../data/dev/nr.gold.dev",
                 "../data/dev/ss.gold.dev"]

    train_words, train_tags, word2index, tag2index, index2tag = read_train_datasets(train_paths)
    dev_words, dev_tags = read_dev_datasets(dev_paths, word2index, tag2index)

    grid_search()