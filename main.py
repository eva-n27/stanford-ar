# coding: utf-8
import numpy as np
import theano
import theano.tensor as T
import lasagne

import sys
import time
import utils
import config
import logging
import nn_layers


def gen_examples(x1, x2, x3, x4, x5, x6, y, batch_size):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t] for t in minibatch]
        mb_x4 = [x4[t] for t in minibatch]
        mb_x5 = [x5[t] for t in minibatch]
        mb_x6 = [x6[t] for t in minibatch]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2)
        mb_x3, mb_mask3 = utils.prepare_data(mb_x3)
        mb_x4, mb_mask4 = utils.prepare_data(mb_x4)
        mb_x5, mb_mask5 = utils.prepare_data(mb_x5)
        mb_x6, mb_mask6 = utils.prepare_data(mb_x6)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3,
                       mb_x4, mb_mask4, mb_x5, mb_mask5, mb_x6, mb_mask6, mb_y))
    return all_ex


def build_fn(args, embeddings):
    """
        Build training and testing functions.
    """
    in_x1 = T.imatrix('x1')
    in_x2 = T.imatrix('x2')
    in_x3 = T.imatrix('x3')  # A
    in_x4 = T.imatrix('x4')  # B
    in_x5 = T.imatrix('x5')  # C
    in_x6 = T.imatrix('x6')  # D

    in_mask1 = T.matrix('mask1')
    in_mask2 = T.matrix('mask2')
    in_mask3 = T.matrix('mask3')
    in_mask4 = T.matrix('mask4')
    in_mask5 = T.matrix('mask5')
    in_mask6 = T.matrix('mask6')

    in_y = T.ivector('y')

    # document
    l_in1 = lasagne.layers.InputLayer((None, None), in_x1)
    l_mask1 = lasagne.layers.InputLayer((None, None), in_mask1)
    l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, args.vocab_size,
                                           args.embedding_size, W=embeddings)

    # question
    l_in2 = lasagne.layers.InputLayer((None, None), in_x2)
    l_mask2 = lasagne.layers.InputLayer((None, None), in_mask2)
    l_emb2 = lasagne.layers.EmbeddingLayer(l_in2, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    # answer A
    l_in3 = lasagne.layers.InputLayer((None, None), in_x3)
    l_mask3 = lasagne.layers.InputLayer((None, None), in_mask3)
    l_emb3 = lasagne.layers.EmbeddingLayer(l_in3, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    # answer B
    l_in4 = lasagne.layers.InputLayer((None, None), in_x4)
    l_mask4 = lasagne.layers.InputLayer((None, None), in_mask4)
    l_emb4 = lasagne.layers.EmbeddingLayer(l_in4, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    # answer C
    l_in5 = lasagne.layers.InputLayer((None, None), in_x5)
    l_mask5 = lasagne.layers.InputLayer((None, None), in_mask5)
    l_emb5 = lasagne.layers.EmbeddingLayer(l_in5, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    # answer D
    l_in6 = lasagne.layers.InputLayer((None, None), in_x6)
    l_mask6 = lasagne.layers.InputLayer((None, None), in_mask6)
    l_emb6 = lasagne.layers.EmbeddingLayer(l_in6, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    # document
    network1 = nn_layers.stack_rnn(l_emb1, l_mask1, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=(args.att_func == 'last'),
                                   bidir=args.bidir,
                                   name='d',
                                   rnn_layer=args.rnn_layer)

    # question
    network2 = nn_layers.stack_rnn(l_emb2, l_mask2, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='q',
                                   rnn_layer=args.rnn_layer)

    # answer A
    network3 = nn_layers.stack_rnn(l_emb3, l_mask3, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='a1',
                                   rnn_layer=args.rnn_layer)

    # answer B
    network4 = nn_layers.stack_rnn(l_emb4, l_mask4, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='a2',
                                   rnn_layer=args.rnn_layer)

    # answer C
    network5 = nn_layers.stack_rnn(l_emb5, l_mask5, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='a3',
                                   rnn_layer=args.rnn_layer)

    # answer D
    network6 = nn_layers.stack_rnn(l_emb6, l_mask6, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='a4',
                                   rnn_layer=args.rnn_layer)

    args.rnn_output_size = args.hidden_size * 2 if args.bidir else args.hidden_size

    answer = lasagne.layers.ConcatLayer([network3, network4, network5, network6], axis=1)
    answer = lasagne.layers.ReshapeLayer(answer, (-1, 4, args.rnn_output_size))

    if args.att_func == 'mlp':
        att = nn_layers.MLPAttentionLayer([network1, network2], args.rnn_output_size,
                                          mask_input=l_mask1)
    elif args.att_func == 'bilinear':
        att = nn_layers.BilinearAttentionLayer([network1, network2], args.rnn_output_size,
                                               mask_input=l_mask1)
    elif args.att_func == 'avg':
        att = nn_layers.AveragePoolingLayer(network1, mask_input=l_mask1)
    elif args.att_func == 'last':
        att = network1
    elif args.att_func == 'dot':
        att = nn_layers.DotProductAttentionLayer([network1, network2], mask_input=l_mask1)
    else:
        raise NotImplementedError('att_func = %s' % args.att_func)

    answer_att = nn_layers.BilinearAttentionLayer([answer, att], args.rnn_output_size)
    network = lasagne.layers.DenseLayer(answer_att, args.num_labels,
                                        nonlinearity=lasagne.nonlinearities.softmax)

    if args.pre_trained is not None:
        dic = utils.load_params(args.pre_trained)
        lasagne.layers.set_all_param_values(network, dic['params'], trainable=True)
        del dic['params']
        logging.info('Loaded pre-trained model: %s' % args.pre_trained)
        for dic_param in dic.iteritems():
            logging.info(dic_param)

    logging.info('#params: %d' % lasagne.layers.count_params(network, trainable=True))
    for layer in lasagne.layers.get_all_layers(network):
        logging.info(layer)

    # Test functions
    test_prob = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.argmax(test_prob, axis=-1)
    acc = T.sum(T.eq(test_prediction, in_y))
    test_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3,
                               in_x4, in_mask4, in_x5, in_mask5, in_x6, in_mask6, in_y], acc)

    # Train functions
    train_prediction = lasagne.layers.get_output(network)
    train_prediction = train_prediction / \
        train_prediction.sum(axis=1).reshape((train_prediction.shape[0], 1))
    train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, in_y).mean()
    # TODO: lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    params = lasagne.layers.get_all_params(network, trainable=True)

    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, params, args.learning_rate)
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(loss, params)
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params)
    else:
        raise NotImplementedError('optimizer = %s' % args.optimizer)
    train_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3,
                               in_x4, in_mask4, in_x5, in_mask5, in_x6, in_mask6, in_y],
                               loss, updates=updates)

    return train_fn, test_fn, params


def eval_acc(test_fn, all_examples):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = 0
    n_examples = 0
    for x1, mask1, x2, mask2, x3, mask3, x4, mask4, x5, mask5, x6, mask6, y in all_examples:
        acc += test_fn(x1, mask1, x2, mask2, x3, mask3, x4, mask4, x5, mask5, x6, mask6, y)
        n_examples += len(x1)
    return acc * 100.0 / n_examples


def main(args):
    logging.info('-' * 50)
    logging.info('Load data files..')

    if args.debug:
        logging.info('*' * 10 + ' Train')
        # 返回的是(documents, questions, answers)
        train_examples = utils.load_data(args.train_file, args.dataset)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.dataset)
    else:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, args.dataset)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.dataset)
    
    args.num_train = len(train_examples[0])
    args.num_dev = len(dev_examples[0])

    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dict = utils.build_dict(args.word_count_file)
    # entity_markers = list(set([w for w in word_dict.keys()
    #                           if w.startswith('@entity')] + train_examples[2]))
    # entity_markers = ['<unk_entity>'] + entity_markers
    # entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
    # logging.info('Entity markers: %d' % len(entity_dict))
    args.num_labels = 4

    logging.info('-' * 50)
    # Load embedding file
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    logging.info('Compile functions..')
    train_fn, test_fn, params = build_fn(args, embeddings)
    logging.info('Done.')

    logging.info('-' * 50)
    logging.info(args)

    logging.info('-' * 50)
    logging.info('Intial test..')
    # 把开发集采样出来的数据向量化
    dev_x1, dev_x2, dev_x3, dev_x4, dev_x5, dev_x6, dev_y = utils.vectorize(dev_examples)
    assert len(dev_x1) == args.num_dev
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_x4, dev_x5, dev_x6, dev_y, args.batch_size)
    dev_acc = eval_acc(test_fn, all_dev)
    logging.info('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc

    if args.test_only:
        return

    utils.save_params(args.model_file, params, epoch=0, n_updates=0)

    # Training
    logging.info('-' * 50)
    logging.info('Start training..')
    train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_y = utils.vectorize(train_examples)
    assert len(train_x1) == args.num_train
    start_time = time.time()
    n_updates = 0

    all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_y, args.batch_size)
    for epoch in range(args.num_epoches):
        np.random.shuffle(all_train)
        for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_x4, mb_mask4, mb_x5, mb_mask5, mb_x6, mb_mask6,
                  mb_y) in enumerate(all_train):
            logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
            train_loss = train_fn(mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_x4, mb_mask4,
                                  mb_x5, mb_mask5, mb_x6, mb_mask6, mb_y)
            logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' %
                         (epoch, idx, len(all_train), train_loss, time.time() - start_time))
            n_updates += 1

            if n_updates % args.eval_iter == 0:
                samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_x1[k] for k in samples],
                                            [train_x2[k] for k in samples],
                                            [train_x3[k] for k in samples],
                                            [train_x4[k] for k in samples],
                                            [train_x5[k] for k in samples],
                                            [train_x6[k] for k in samples],
                                            [train_y[k] for k in samples],
                                            args.batch_size)
                logging.info('Train accuracy: %.2f %%' % eval_acc(test_fn, sample_train))
                dev_acc = eval_acc(test_fn, all_dev)
                logging.info('Dev accuracy: %.2f %%' % dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logging.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, dev_acc))
                    utils.save_params(args.model_file, params, epoch=epoch, n_updates=n_updates)


if __name__ == '__main__':
    args = config.get_args()
    np.random.seed(args.random_seed)
    lasagne.random.set_rng(np.random.RandomState(args.random_seed))

    if args.train_file is None:
        raise ValueError('train_file is not specified.')

    if args.dev_file is None:
        raise ValueError('dev_file is not specified.')

    if args.rnn_type == 'lstm':
        args.rnn_layer = lasagne.layers.LSTMLayer
    elif args.rnn_type == 'gru':
        args.rnn_layer = lasagne.layers.GRULayer
    else:
        raise NotImplementedError('rnn_type = %s' % args.rnn_type)

    if args.embedding_file is not None:
        dim = utils.get_dim(args.embedding_file)
        if (args.embedding_size is not None) and (args.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                             (args.embedding_size, args.embedding_file, dim))
        args.embedding_size = dim
    elif args.embedding_size is None:
        raise RuntimeError('Either embedding_file or embedding_size needs to be specified.')

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.info(' '.join(sys.argv))
    main(args)
