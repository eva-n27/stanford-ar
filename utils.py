# coding: utf-8
import os
import numpy as np
import lasagne
import config
import cPickle as pickle
import gzip
import logging


def load_data(file_address_, dataset_):
    """
    load data from train/dev file
    :return:
    """
    context_ = []
    question_ = []
    answer_ = []
    answer_1_ = []
    answer_2_ = []
    answer_3_ = []
    answer_4_ = []

    if dataset_ == 'gaokao':
        for file_name_ in os.listdir(file_address_):
            with open(file_address_ + '/' + file_name_, 'r') as f_:
                lines_ = f_.readlines()
                # gold answer
                try:
                    answer_.append(int(lines_[-1].strip()))  # remove '\n'
                except ValueError:
                    print file_name_
                lines_.pop()

                # answer list
                a_lst_ = []
                for i_ in range(4):
                    line_ = lines_[-1].strip().split()
                    for j_ in range(len(line_)):
                        line_[j_] = int(line_[j_])
                    a_lst_.append(line_)
                    lines_.pop()
                answer_1_.append(a_lst_[3])
                answer_2_.append(a_lst_[2])
                answer_3_.append(a_lst_[1])
                answer_4_.append(a_lst_[0])

                # question
                line_ = lines_[-1].strip().split()
                for i_ in range(len(line_)):
                    line_[i_] = int(line_[i_])
                question_.append(line_)
                lines_.pop()

                # context
                c_ = []
                # 这个时候,lines中只剩下了context了
                for i_ in range(len(lines_)):
                    line_ = lines_[i_].strip().split()
                    for j_ in range(len(line_)):
                        line_[j_] = int(line_[j_])
                    c_.extend(line_)
                context_.append(c_)

    elif dataset_ == 'race':
        for file_name_ in os.listdir(file_address_):
            with open(file_address_ + '/' + file_name_, 'r') as f_:
                lines_ = f_.readlines()

                # read context
                line_ = lines_[0].strip().split()
                for i_ in range(len(line_)):
                    line_[i_] = int(line_[i_])
                context_.append(line_)

                # read question
                line_ = lines_[1].strip().split()
                for i_ in range(len(line_)):
                    line_[i_] = int(line_[i_])
                question_.append(line_)

                # read answer list
                a_l_ = []
                for i_ in range(2, 6):
                    line_ = lines_[i_].strip().split()
                    for j_ in range(len(line_)):
                        line_[j_] = int(line_[j_])
                    a_l_.append(line_)
                answer_1_.append(a_l_[0])
                answer_2_.append(a_l_[1])
                answer_3_.append(a_l_[2])
                answer_4_.append(a_l_[3])

                # read gold answer
                answer_.append(int(lines_[6].strip().split()[0]))

    return context_, question_, answer_1_, answer_2_, answer_3_, answer_4_, answer_


def build_dict(word_count_file):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    f = open(word_count_file, 'r')
    word_count = pickle.load(f)
    f.close()

    ls = word_count.most_common()
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    # leave 1 to UNK
    # 因为我处理过的数据中UNK是1,所以我这里还是用的1,0没有用到,后续看代码的时候再看看吧
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}


def vectorize(examples, sort_by_len=True):
    """
        examples: {context, question, answer list, gold answer}
        因为数据已经向量化过,因此直接进行向量化之后的操作
        in_x1, in_x2, in_x3: sequences for document, question and answer list respecitvely.
        in_y: lable
    """
    in_x1 = examples[0]
    in_x2 = examples[1]
    in_x3 = examples[2]
    in_x4 = examples[3]
    in_x5 = examples[4]
    in_x6 = examples[5]
    in_y = examples[6]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_x3 = [in_x3[i] for i in sorted_index]
        in_x4 = [in_x4[i] for i in sorted_index]
        in_x5 = [in_x5[i] for i in sorted_index]
        in_x6 = [in_x6[i] for i in sorted_index]
        in_y = [in_y[i] for i in sorted_index]

    return in_x1, in_x2, in_x3, in_x4, in_x5, in_x6, in_y


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype(config._floatX)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def get_dim(in_file):
    line = open(in_file).readline()
    return len(line.split()) - 1


def gen_embeddings(word_dict, dim, in_file=None,
                   init=lasagne.init.Uniform()):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = max(word_dict.values()) + 1
    embeddings = init((num_words, dim))
    logging.info('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file).readlines():
            sp = line.split()
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        logging.info('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


def save_params(file_name, params, **kwargs):
    """
        Save params to file_name.
        params: a list of Theano variables
    """
    dic = {'params': [x.get_value() for x in params]}
    dic.update(kwargs)
    with gzip.open(file_name, "w") as save_file:
        pickle.dump(obj=dic, file=save_file, protocol=-1)


def load_params(file_name):
    """
        Load params from file_name.
    """
    with gzip.open(file_name, "rb") as save_file:
        dic = pickle.load(save_file)
    return dic
