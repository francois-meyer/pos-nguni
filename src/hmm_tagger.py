import re
import sys
import math
import string
from sklearn.metrics import f1_score


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
            # if gold_tags[i][j] == tag2index["<pad>"]:
            #     break

            total += 1
            if pred_tags[i][j] == gold_tags[i][j]:
                correct += 1

        seq_len = len(gold_tags[i])

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


def interpl(bigram_p, unigram_p):
    factor = 0.6
    return -math.log(factor * math.exp(-bigram_p) + (1 - factor) * math.exp(-unigram_p))


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

    print('read training data')


def read_train_datasets(dataset_paths):
    word2index = {}
    tag2index = {}
    word2index['<s>'] = 0
    word2index['<unk>'] = 1
    tag2index['<s>'] = 0
    index2tag = ['<s>']

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

        word, morphs, lemma, xpos, tag = line.rsplit("\t", maxsplit=4)
        if word == ".":
            new_sent = True
        else:
            if new_sent:
                words.append([0])
                tags.append([0])
                new_sent = False


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


if __name__ == '__main__':

    train_paths = ["../data/train/ss.gold.train"]
    dev_paths = ["../data/gold/ss.gold.test"]

    train_words, train_tags, word2index, tag2index, index2tag = read_train_datasets(train_paths)
    print(len(train_words))

    num_tags = len(tag2index)
    transition_counts = []  # tag-> tag bigram counts
    trigram_counts = []  # tag -> tag -> tag trigram counts

    # Add smoothing constant for bigram counts and initialise trigram count dicts
    for k in range(num_tags):
        transition_counts.append([1] * num_tags)  # plus one smoothing
        trigram_counts.append([])
        for l in range(num_tags):
            trigram_counts[-1].append({})

    # Go through tag corpus and compute bigram and trigram counts
    # In the end transition_counts[i][j] has the count for word i -> word j
    for sent in train_tags:
        for i in range(1, len(sent)):
            transition_counts[sent[i - 1]][sent[i]] += 1
        for i in range(2, len(sent)):
            if sent[i] in trigram_counts[sent[i - 2]][sent[i - 1]]:
                trigram_counts[sent[i - 2]][sent[i - 1]][sent[i]] += 1
            else:
                trigram_counts[sent[i - 2]][sent[i - 1]][sent[i]] = 1

    # Go through corpus and collect tag->word emission counts and word unigram counts
    unigram_emm_prob = [0.0] * len(word2index)  # unigram word probabilities
    emmision_counts = []  # tag->word emission counts as list of dicts [tag0 {word0: count, word1: count}, tag1 ...]
    for k in range(num_tags):
        emmision_counts.append({1: 1})  # unk emmision has count 1
    unigram_emm_prob[1] = num_tags

    for j, sent in enumerate(train_tags):
        for i in range(len(sent)):
            word = train_words[j][i]
            tag = train_tags[j][i]
            unigram_emm_prob[word] += 1
            if word in emmision_counts[tag]:
                emmision_counts[tag][word] += 1
            else:
                emmision_counts[tag][word] = 1
    emm_sum = sum(unigram_emm_prob)
    for j in range(len(unigram_emm_prob)):
        unigram_emm_prob[j] = -math.log(unigram_emm_prob[j]) + math.log(emm_sum)  # log unigram probabilities

    # negative log probabilities
    unigram_prob = []  # unigram tag probs
    transition_prob = []  # tag->tag transition probs
    trigram_prob = []  # tag->tag->tag transition probs
    for j in range(num_tags):
        transition_prob.append([sys.maxsize] * num_tags)
        trigram_prob.append([])
        tag_sum = sum(transition_counts[j]) + 0.0
        unigram_prob.append(tag_sum)
        for i in range(num_tags):
            transition_prob[j][i] = -math.log(transition_counts[j][i]) + math.log(tag_sum)
            trigram_prob[-1].append({})
            for k in range(num_tags):
                if k in trigram_counts[j][i]:
                    trigram_prob[j][i][k] = -math.log(trigram_counts[j][i][k]) + math.log(
                        transition_counts[j][i] - 1)  # exact prob

    # Compute tag unigram log probs
    tag_sum = sum(unigram_prob) + 0.0
    for j in range(len(unigram_prob)):
        unigram_prob[j] = -math.log(unigram_prob[j]) + math.log(tag_sum)

    emmision_prob = []  # tag->word emission log probs
    for k in range(num_tags):
        emmision_prob.append({})
        emm_sum = sum(emmision_counts[k].values()) + 0.0
        for i in emmision_counts[k]:
            emmision_prob[k][i] = -math.log(emmision_counts[k][i]) + math.log(emm_sum)

    print("trained model")

    dev_words, dev_tags = read_dev_datasets(dev_paths)

    num_test_tokens = 0
    num_correct_tokens = 0
    index2word = {v: k for k, v in word2index.items()}

    pred_tags = []
    gold_tags = []

    # viterbi decoding
    for j, sent in enumerate(dev_words):
        forward_prob = []
        back_index = []
        # forward pass
        for i, word in enumerate(sent):
            forward_prob.append([sys.maxsize] * num_tags)
            back_index.append([None] * num_tags)
            if i == 0:
                forward_prob[i][0] = 0
                continue
            for tag in range(num_tags):
                if word in emmision_prob[tag]:
                    # can do interpolation during training
                    # forward_prob[i][tag], back_index[i][tag] = min((forward_prob[i-1][prev_tag] + transition_prob[prev_tag][tag] + interpl(emmision_prob[tag][word], unigram_emm_prob[word]), prev_tag) for prev_tag in range(num_tags))
                    forward_prob[i][tag], back_index[i][tag] = min((forward_prob[i - 1][prev_tag] +
                                                                    transition_prob[prev_tag][tag] + emmision_prob[tag][
                                                                        word], prev_tag) for prev_tag in
                                                                   range(num_tags))
                    # forward_prob[i][tag], back_index[i][tag] = min((forward_prob[i-1][prev_tag] + interpl(transition_prob[prev_tag][tag], trigram_prob[back_index[i-1][prev_tag] or 0][prev_tag][tag] or sys.maxsize) + emmision_prob[tag][word], prev_tag) for prev_tag in range(num_tags))

        # trace back best tag sequence
        final_prob, final_tag = min((forward_prob[-1][tag], tag) for tag in range(num_tags))
        best_path = [final_tag]
        for k in range(len(sent) - 1, 0, -1):
            best_path.append(back_index[k][best_path[-1]])
        best_path.reverse()
        num_test_tokens += len(dev_tags[j])

        pred_tags.append(best_path[1:])
        gold_tags.append(dev_tags[j][1:])
        for i in range(len(dev_tags[j])):
            this_word = index2word[sent[i]]
            if this_word == "<s>":
                num_test_tokens -= 1
                continue

            if best_path[i] == dev_tags[j][i]:
                num_correct_tokens += 1



    acc = (num_correct_tokens + 0.0) / num_test_tokens
    print(acc)
    acc, f1 = compute_scores(gold_tags, pred_tags, tag2index)
    acc *= 100
    f1 *= 100
    print("{:.2f}".format(acc), "{:.2f}".format(f1))

