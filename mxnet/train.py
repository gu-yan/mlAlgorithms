# -*- coding:utf-8 -*-


import mxnet as mx
import log
import logging
import data_tool
import yaml
import sys
import os
import argparse
from model import resnet
from model import resnext


logging.getLogger().setLevel(logging.DEBUG)
logger = log.get_logger(name='train', filename='train.log', filemode='a', level=log.DEBUG, file_and_line=True)


def process(train_params):
    """

    :param train_params:
    :return:
    """
    logger.info('prepare train data')
    features_train, labels_train, features_test, labels_test = data_tool.load_data(
        train_params['train']['traindatapath'], train_params['train']['testdatapath'],
        one_hot=False)

    train_iter = mx.io.NDArrayIter(data=features_train, label=labels_train,
                                   batch_size=train_params['train']['batchsize'], shuffle=True)
    eval_iter = mx.io.NDArrayIter(data=features_test, label=labels_test,
                                  batch_size=train_params['train']['batchsize'])

    logger.info('load symbol %s' % train_params['train']['net'])

    symbol = None
    if train_params['train']['net'] == 'resnext34':
        symbol = resnext.get_symbol(num_classes=train_params['classes'], num_layers=34,
                                    image_shape=train_params['inputshape'])
    elif train_params['train']['net'] == 'resnext50':
        symbol = resnext.get_symbol(num_classes=train_params['classes'], num_layers=50,
                                    image_shape=train_params['inputshape'])
    elif train_params['train']['net'] == 'resnext101':
        symbol = resnext.get_symbol(num_classes=train_params['classes'], num_layers=101,
                                    image_shape=train_params['inputshape'])
    elif train_params['train']['net'] == 'resnet34':
        symbol = resnext.get_symbol(num_classes=train_params['classes'], num_layers=34,
                                    image_shape=train_params['inputshape'])
    elif train_params['train']['net'] == 'resnet50':
        symbol = resnext.get_symbol(num_classes=train_params['classes'], num_layers=50,
                                    image_shape=train_params['inputshape'])
    elif train_params['train']['net'] == 'resnet101':
        symbol = resnext.get_symbol(num_classes=train_params['classes'], num_layers=101,
                                    image_shape=train_params['inputshape'])
    elif train_params['train']['net'] == 'resnet152':
        symbol = resnext.get_symbol(num_classes=train_params['classes'], num_layers=152,
                                    image_shape=train_params['inputshape'])

    logger.info('create model graph at %s' % train_params['train']['graphpath'])
    mx.viz.plot_network(symbol).render(train_params['train']['graphpath'])

    model = mx.mod.Module(symbol=symbol, logger=logger, context=mx.gpu(0))

    logger.info('begin to train==============, epoch: %d, optimizer: %s, learning_rate: %d, batchsize: %d'
                % (train_params['train']['trainepoch'], train_params['train']['optimizer'],
                    train_params['train']['learningrate'], train_params['train']['batchsize']))

    if train_params['preimagnettrain']:
        sym, arg_params, aux_params = get_imagenet_model(train_params['train']['net'])
    if train_params['pretrain']:
        sym, arg_params, aux_params = get_trainned_model(train_params['train']['modelprefix'])

    model.fit(train_data=train_iter,
              eval_data=eval_iter,
              arg_params=arg_params,
              aux_params=aux_params,
              eval_metric=train_params['train']['evalmetric'],
              optimizer=train_params['train']['optimizer'],
              optimizer_params={'learning_rate': train_params['train']['learningrate']},
              epoch_end_callback=mx.callback.do_checkpoint(prefix=train_params['train']['modelprefix'],
                                                           period=train_params['train']['modelepoch']),
              batch_end_callback=mx.callback.Speedometer(batch_size=train_params['train']['batchsize'],
                                                         frequent=1),
              num_epoch=train_params['train']['trainepoch'])

    metric = mx.metric.Accuracy()
    model.score(eval_iter, metric)
    logger.info('train end============== score %f' % metric.get()[1])


def get_imagenet_model(net_name):
    model_prefix = None
    if net_name == 'resnext50':
        model_prefix = 'imagenet/resnext-50'
    elif net_name == 'resnext101':
        model_prefix = 'imagenet/resnext-101'
    elif net_name == 'resnet34':
        model_prefix = 'imagenet/resnet-34'
    elif net_name == 'resnet50':
        model_prefix = 'imagenet/resnet-50'
    elif net_name == 'resnet101':
        model_prefix = 'imagenet/resnet-101'
    elif net_name == 'resnet152':
        model_prefix = 'imagenet/resnet-152'
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
    return sym, arg_params, aux_params


def get_trainned_model(model_prefix):
    if model_prefix:
        if model_prefix.rindex('/') > 0:
            index = model_prefix[:model_prefix.rindex('/')]
            file_list = os.listdir(index)
        else:
            file_list = os.listdir('')
        params_file_list = []
        for item in file_list:
            if item.endswith('.params'):
                params_file_list.append(item)
        params_file_list.sort(reverse=True)
        if len(params_file_list) > 0:
            param_file = params_file_list[0]
            from_epoch = int(param_file[param_file.rindex('-')+1:param_file.rindex('.')])
            sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix,
                                                                   from_epoch)
            logger.info('load pretrainned model from %s %d' % (model_prefix, from_epoch))
            return sym, arg_params, aux_params
    return None, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train input',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--parampath', type=str, default='params.yaml',
                        help='path of params.yaml')
    args = parser.parse_args()

    logger.info('load params from %s' % args.parampath)

    params = None
    with open(args.parampath) as f:
        params = yaml.load(f)

    if params is None:
        logger.error('no params found from %s' % args.parampath)
        sys.exit(1)

    process(params)
