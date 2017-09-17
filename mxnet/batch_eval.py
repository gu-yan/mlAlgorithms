# -*- coding:utf-8 -*-


import argparse
import yaml
import sys
import eval
import log
import os


logger = log.get_logger(name='eval', filename='eval.log', filemode='a', level=log.DEBUG, file_and_line=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modelprefix', type=str,
                        help='prefix of the *.params')
    parser.add_argument('--epoch', type=int, default=0,
                        help='Epoch number of trained model we would like to load')
    parser.add_argument('--imagepath', type=str, default='image',
                        help='image absolute path')
    parser.add_argument('--parampath', type=str, default='params.yaml',
                        help='path of params.yaml')
    parser.add_argument('--format', type=str, default='NCHW',
                        help='format of image array. default is NCHW for mxnet conv, NHWC is supported')
    args = parser.parse_args()

    logger.info('load params from %s' % args.parampath)

    params = None
    with open(args.parampath) as f:
        params = yaml.load(f)

    if params is None:
        logger.error('no params found from %s' % args.parampath)
        sys.exit(1)

    files = os.listdir(args.imagepath)

    images = []
    for _ in files:
        images.append(os.path.join(args.imagepath, _))

    evaer = eval.Eval(modelprefix=args.modelprefix,
                      imagepath=images,
                      inputshape=params['inputshape'],
                      labelpath=params['eval']['label'],
                      epoch=args.epoch,
                      format=args.format)
    names, prob_predicts, label_predicts = evaer.predict()

    reports = []

    for i in range(len(names)):
        print('name: %s, predict: %s, acc: %f' % (names[i], label_predicts[i], prob_predicts[i]))
        reports.append(names[i] + ' : ' + label_predicts[i] + ' : ' + str(prob_predicts[i]) + '\n')

    with open('reports.txt', 'a') as f:
        for report in reports:
            f.write(report)
