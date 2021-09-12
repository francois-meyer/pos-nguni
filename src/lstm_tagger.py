import re
import sys
import math
import string
import torch
from torch import nn, optim
import time
from torch.nn.utils.rnn import pad_sequence


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
        else:
            if new_sent:

                # if len(words) > 0:
                #     index2word = {v: k for k, v in word2index.items()}
                #     sentence = " ".join([index2word[idx] for idx in words[-1]])
                #     print(sentence)

                words.append([0])
                tags.append([0])
                new_sent = False

            if word == "" and tag != "":
                word = "."

            if not word in word2index:
                word2index[word] = len(word2index)
            words[-1].append(word2index[word])
            if not tag in tag2index:
                tag2index[tag] = len(tag2index)
                index2tag.append(tag)
            tags[-1].append(tag2index[tag])



def read_train_datasets(dataset_paths):
    word2index = {}
    tag2index = {}
    word2index["<s>"] = 0
    word2index["<unk>"] = 1
    word2index["<pad>"] = 2
    tag2index["<s>"] = 0
    tag2index["<pad>"] = 1
    index2tag = ["<s>", "<pad>"]

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

        if line.strip() == "," or line.strip() == "":
            new_sent = True
        else:
            if new_sent:
                words.append([0])
                tags.append([0])
                new_sent = False

            word, morphs, tag = line.rsplit("\t", maxsplit=2)
            if word == "" and tag != "":
                word = "."

            if not word in word2index:  # Changed is to in
                word = '<unk>'
            words[-1].append(word2index[word])
            if not tag in tag2index:
                # print 'unknown gold tag', tag
                tags[-1].append(0)
            else:
                tags[-1].append(tag2index[tag])


def read_dev_datasets(dataset_paths):
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
        embeddings = self.drop(self.embedding(input_ids))
        hidden_states, final_states = self.lstm(embeddings, (init_state_h, init_state_c))
        output = self.drop(hidden_states)
        logits = self.fc(output)
        return logits

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


if __name__ == '__main__':

    train_paths = ["../data/train/xh.gold.train", "../data/train/zu.gold.train", "../data/train/nr.gold.train", "../data/train/ss.gold.train"]
    dev_paths = ["../data/dev/xh.gold.dev", "../data/dev/zu.gold.dev", "../data/dev/nr.gold.dev", "../data/dev/ss.gold.dev"]

    train_words, train_tags, word2index, tag2index, index2tag = read_train_datasets(train_paths)
    dev_words, dev_tags = read_dev_datasets(dev_paths)

    index2word = {v: k for k, v in word2index.items()}
    word_pad_id = word2index["<pad>"]
    tag_pad_id = tag2index["<pad>"]
    num_tags = len(tag2index)

    word_vocab_size = len(word2index)
    tag_vocab_size = len(tag2index)
    input_size = 128
    hidden_size = 256
    num_layers = 1
    dropout = 0.2
    num_epochs = 10
    weight_decay = 1e-5
    lr_patience = 2
    lr = 0.001
    batch_size = 12
    bptt_len = 120
    clip = 1.0
    log_interval = 1
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_id)

    # Set up model and training
    model = LSTMTagger(word_vocab_size, tag_vocab_size, input_size, hidden_size, num_layers, dropout, word_pad_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # Adam with proper weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience, verbose=True,
                                                     factor=0.5)  # Half the learning rate

    # Train model
    start = time.time()
    best_loss = float("inf")
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        total_loss = 0

        for batch_num, batch_pos in enumerate(range(0, len(train_words), batch_size)):

            batch_words = train_words[batch_pos: batch_pos + batch_size]
            batch_words = [torch.tensor(batch_words[i]) for i in range(len(batch_words))]
            batch_words = pad_sequence(batch_words, padding_value=word_pad_id)

            batch_tags = train_tags[batch_pos: batch_pos + batch_size]
            batch_tags = [torch.tensor(batch_tags[i]) for i in range(len(batch_tags))]
            batch_tags = pad_sequence(batch_tags, padding_value=tag_pad_id)

            model.train()
            model.zero_grad()

            init_state_h, init_state_c = model.init_states(batch_size)
            logits = model(batch_words, init_state_h, init_state_c)

            loss = criterion(input=logits.view(-1, num_tags), target=batch_tags.view(-1))
            #nll = - torch.sum(log_alpha) / torch.numel(input_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item()

            if batch_num % log_interval == 0 and batch_num > 0:  # and batch_num > 0:
                cur_loss = total_loss / log_interval
                print("| epoch {:3d} | loss {:5.2f} | ".format(epoch, cur_loss))
                total_loss = 0

        # val_loss = evaluate(model, char_vocab, valid_corpus, device, batch_size, bptt_len, reg_coef)
        # print("| epoch {:3d} | valid loss {:5.2f} | valid R {:5.2f} | valid nll {:5.2f} | valid ppl {:5.2f} | "
        #       "valid bpc {:5.2f} | ".format(epoch, val_loss, val_R, val_nll, math.exp(val_nll), val_nll / math.log(2)))
        # # generate_text(model, vocab, "Tears will gather ", gen_len=100)
        # scheduler.step(val_loss)
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_epoch = epoch

    end = time.time()
    print("Training completed in %fs." % (end - start))
    #print("| epoch {:3d} | valid loss {:5.2f} | ".format(best_epoch, best_loss))


    # Evaluation
    # num_test_tokens = 0
    # num_correct_tokens = 0
    # index2word = {v: k for k, v in word2index.items()}
    #
    # for j, sent in enumerate(dev_words):
    #     for i in range(len(dev_tags[j])):
    #         this_word = index2word[sent[i]]
    #         if all(ch in string.punctuation for ch in this_word) or this_word == "<s>":
    #             num_test_tokens -= 1
    #             continue
    #         if best_path[i] == dev_tags[j][i]:
    #             num_correct_tokens += 1
    #
    # acc = (num_correct_tokens + 0.0) / num_test_tokens
    # print(acc)
