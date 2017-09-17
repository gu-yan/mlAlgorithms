# -*- coding:utf-8 -*-


from PIL import Image
from PIL import ImageEnhance
import numpy as np


def get_image(path, shape, format):
    """

    :param path:
    :param shape:
    :param format:
    :return:
    """
    img = Image.open(path)
    img = img.resize(size=(shape[1], shape[2]), resample=Image.LANCZOS)
    img_array = np.asarray(img, dtype=np.uint8)
    if format == 'NCHW':
        img_array = img_array.transpose(2, 0, 1)
    img_array = img_array.reshape([1, shape[0], shape[1], shape[2]])
    return img_array


class Enhance(object):

    def __init__(self,
                 path=None,
                 img=None):
        self.path = path
        self.img = img

    def set_img(self, img):
        self.img = img

    def color_enhance(self, factor, new_path=None, is_show=False):
        if self.img is None:
            img = Image.open(self.path)
        else:
            img = self.img
        img = ImageEnhance.Color(img).enhance(factor)
        if new_path is not None:
            img.save(new_path)
        if is_show:
            img.show(title='color')
        return img

    def brightness_enhance(self, factor, new_path=None, is_show=False):
        if self.img is None:
            img = Image.open(self.path)
        else:
            img = self.img
        img = ImageEnhance.Brightness(img).enhance(factor)
        if new_path is not None:
            img.save(new_path)
        if is_show:
            img.show(title='brightness')
        return img

    def contrast_enhance(self, factor, new_path=None, is_show=False):
        if self.img is None:
            img = Image.open(self.path)
        else:
            img = self.img
        img = ImageEnhance.Contrast(img).enhance(factor)
        if new_path is not None:
            img.save(new_path)
        if is_show:
            img.show(title='contrast')
        return img

    def sharpness_enhance(self, factor, new_path=None, is_show=False):
        if self.img is None:
            img = Image.open(self.path)
        else:
            img = self.img
        img = ImageEnhance.Sharpness(img).enhance(factor)
        if new_path is not None:
            img.save(new_path)
        if is_show:
            img.show(title='sharpness')
        return img
