# -*- coding:utf-8 -*-


import argparse
import os
import sys
from PIL import Image
import random
import numpy as np


def generate_ndarray():
    """
        your picture should be named like name_labelindex.xx(abc_0.jpg)
    :return:
    """
    file_list = os.listdir(args.imagepath)

    if file_list is None:
        print('empty path')
        sys.exit(1)

    file_list.sort()
    if args.shuffle:
        random.seed(100)
        random.shuffle(file_list)

    file_lists = slice(file_list)

    for _ in range(len(file_lists)):
        part_data_list = []
        part_lable_list = []
        for item in file_lists[_]:
            # result is (width, height) RGB
            image = Image.open(os.path.join(args.imagepath, item))
            # get lable index
            part_lable_list.append(int(item[item.rindex('_') + 1:item.rindex('.')]))

            if args.channel == 1:
                # conver to grey image
                image = image.convert('L')
            if image.size[0] != imgsize[1] or image.size[1] != imgsize[0]:
                # resize image
                image = image.resize(size=(imgsize[1], imgsize[0]), resample=Image.LANCZOS)

            img_array = np.asarray(image, dtype=np.uint8)

            # save as format: NCHW
            if args.format == 'NCHW':
                img_array = img_array.transpose(2, 0, 1)

            part_data_list.append(img_array)

        data = {
            'data': np.asarray(part_data_list),
            'label': np.asarray(part_lable_list)
        }

        with open(os.path.join(args.datapath, 'part_' + str(_)), 'wb') as fb:
            major = sys.version_info.major
            if 2 == major:
                import cPickle
                cPickle.dump(data, fb, 1)
            elif 3 == major:
                import pickle
                pickle.dump(data, fb, 1)


def slice(image_list):
    res = []
    if args.slice >= 2:
        if len(image_list) < args.slice:
            print('image size is smaller than slice')
            sys.exit(1)
        each = int(len(image_list)/args.slice)
        for _ in range(args.slice):
            if _ == args.slice - 1:
                res.append(image_list[_ * each:])
            else:
                res.append(image_list[_ * each:(_ + 1) * each])
    else:
        res.append(image_list)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert iamge to typical format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--imagepath', type=str, default='image',
                        help='path of iamges')
    parser.add_argument('--datapath', type=str, default='data',
                        help='path of generated file')
    parser.add_argument('--imgsize', type=str, default='512, 512',
                        help='shape of new image, (height, width)')
    parser.add_argument('--channel', type=int, default=3,
                        help='channel of image, default is 3,if set to 1 then image will be convert to grey')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='shuffle the image')
    parser.add_argument('--slice', type=int, default=2,
                        help='slice the data to serveral part')
    parser.add_argument('--format', type=str, default='NCHW',
                        help='format of image array. default is NCHW for mxnet conv, NHWC is supported')

    args = parser.parse_args()

    if not os.path.exists(args.datapath):
        os.mkdir(args.datapath)

    imgsize = [int(l) for l in args.imgsize.split(',')]
    generate_ndarray()
