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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    word2index["<pad>"] = 0
    word2index["<unk>"] = 1
    # word2index["<s>"] = 2

    tag2index = {}
    tag2index["<pad>"] = 0
    tag2index["<unk>"] = 1
    tag2index[START_TAG] = 2
    tag2index[STOP_TAG] = 3

    # tag2index["<s>"] = 2
    index2tag = ["<unk", "<pad>", START_TAG, STOP_TAG]  # "<s>"

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


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

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
    model = BiLSTM_CRF_Tagger(vocab_size=word_vocab_size, tagset_size=num_tags, embedding_dim=input_size,
                              hidden_dim=hidden_size, num_rnn_layers=num_layers)
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

            loss = model.loss(batch_words.T, batch_tags.T)

            #loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
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

                logits, tag_seq = model(batch_words.T)
                batch_pred_tags = torch.max(logits, dim=-1).indices.T
                #tag_seq_T = np.array([np.array(seq) for seq in tag_seq])
                #tag_seq_T = tag_seq_T.T
                #tag_seq_T = tag_seq_T.tolist()
                pred_tags.extend(tag_seq)

                #loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
                #val_loss += loss.item()

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

import torch
import torch.nn as nn


def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from features space to tag space.
    :param in_features: number of features for the input
    :param num_tag: number of tags. DO NOT include START, STOP tags, they are included internal.
    """

    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        features = self.fc(features)
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        features = self.fc(features)

        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence
        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores


class BiLSTM_CRF_Tagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, num_rnn_layers=1, rnn="lstm"):
        super(BiLSTM_CRF_Tagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(embedding_dim, hidden_dim // 2, num_layers=num_rnn_layers,
                       bidirectional=True, batch_first=True)
        self.crf = CRF(hidden_dim, self.tagset_size)

    def __build_features(self, sentences):
        masks = sentences.gt(0)
        embeds = self.embedding(sentences.long())

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks

    def loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq


if __name__ == '__main__':
    params = {}
    params["input_size"] = 512
    params["hidden_size"] = 512
    params["num_layers"] = 2
    params["dropout"] = 0.2
    params["num_epochs"] = 50
    params["weight_decay"] = 1e-5
    params["lr_patience"] = 3
    params["lr"] = 0.001
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


