# -*- coding:utf-8 -*-


import log
import mxnet as mx
import numpy as np
from collections import namedtuple
from cv_tools import image_tool


logger = log.get_logger(name='eval', filename='eval.log', filemode='a', level=log.DEBUG, file_and_line=True)


class Eval:

    def __init__(self,
                 modelprefix,
                 imagepath,
                 inputshape,
                 labelpath,
                 epoch=0,
                 format='NCHW'):
        self.modelprefix = modelprefix
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.inputshape = inputshape
        self.epoch = epoch
        self.format = format

        with open(labelpath, 'r') as fo:
            self.labels = [l.rstrip() for l in fo]

        sym, arg_params, aux_params = mx.model.load_checkpoint(self.modelprefix, self.epoch)
        self.mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
        self.mod.bind(for_training=False,
                      data_shapes=[('data', (1, self.inputshape[0], self.inputshape[1], self.inputshape[2]))],
                      label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)

    def predict(self):
        Batch = namedtuple('Batch', ['data'])

        datas = []
        names = []
        prob_predicts = []
        label_predicts = []

        logger.info('begin to load image')
        for _ in self.imagepath:
            x = image_tool.get_image(_, self.inputshape, format=self.format)
            datas.append(x)
            names.append(_)

        logger.info('begin to predict')
        for _ in datas:
            self.mod.forward(Batch([mx.nd.array(_)]))
            prob = self.mod.get_outputs()[0].asnumpy()

            # print the top-1
            prob = np.squeeze(prob)
            a = np.argsort(prob)[::-1]
            for i in a[0:1]:
                print('probability=%f, class=%s' % (prob[i], self.labels[i]))
                prob_predicts.append(prob[i])
                label_predicts.append(self.labels[i])
        return names, prob_predicts, label_predicts
