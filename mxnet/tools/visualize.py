# -*- coding:utf-8 -*-


import argparse
import matplotlib.pyplot as plt


def view():
    result = []
    xindex = []
    index = 0
    for line in open(args.logpath):
        if 'Validation-accuracy' in line:
            result.append(line[line.index('=') + 1:])
            xindex.append(index)
            index += 1
    if len(result) >= 1:
        plt.title('')
        plt.xlabel('epoch')
        plt.ylabel('res')
        plt.plot(xindex, result)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize log',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logpath', type=str, default='log.log',
                        help='log path')

    args = parser.parse_args()
    view()
