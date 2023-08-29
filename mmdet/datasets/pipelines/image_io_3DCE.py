import cv2
import os
import os.path as osp

import numpy
import numpy as np
from PIL import Image
SLICE_INTERVAL = 2.5
import pdb

def load_multislice_png(imname, num_slice=9, window=None):
    # Load single channel image for lesion detection
    def get_slice_name(img_path, delta=0):
        if delta == 0:
            return img_path
        delta = int(delta)
        name_slice_list = img_path.split(os.sep)
        slice_idx = int(name_slice_list[-1][-7:-4])
        #slice_idx = name_slice_list[8]
        #slice_idx = int(name_slice_list[-5][:-7])
        img_name =  name_slice_list[-1][:-7]+'%03d.png' % (slice_idx + delta)
        full_path = os.path.join('/', *name_slice_list[:-1], img_name)

        # if the slice is not in the dataset, use its neighboring slice
        while not os.path.exists(full_path):
            # print('file not found:', img_name)
            delta -= np.sign(delta)
            img_name =  name_slice_list[-1][:-7]+'%03d.png' % (slice_idx + delta)
            full_path = os.path.join('/', *name_slice_list[:-1], img_name)
            if delta == 0:
                break
        return full_path

    def _load_data(img_name, delta=0):  
        img_name = get_slice_name(img_name, delta)
        if img_name not in data_cache.keys():
            data_cache[img_name] = cv2.imread(img_name, -1)  #-1代表8位深度，原通道；0代表8位深度，1通道（灰度图）
            # temp = numpy.array(Image.open(img_name))
            # if temp.shape[2] == 4:
            #    temp = temp[:, :, :3]
            # data_cache[img_name] = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
            if data_cache[img_name] is None:
                print('file reading error:', img_name, os.path.exists(img_name))
                assert not data_cache[img_name] == None
        return data_cache[img_name]

    def _load_multi_data(im_cur, imname, num_slice):
        ims = [im_cur]
        
        for p in range((num_slice - 1) // 2):
            intv1 = p + 1
            slice1 = _load_data(imname, - intv1)
            im_prev = slice1
    
            slice2 = _load_data(imname, intv1)
            im_next = slice2
            ims = [im_prev] + ims + [im_next]
        # when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
        if num_slice % 2 == 0:
            ntv1 = p + 2
            slice1 = _load_data(imname, np.ceil(intv1))
            slice2 = _load_data(imname, np.floor(intv1))
            im_next = slice1 + slice2
            ims += [im_next]
        

        return ims
    
    data_cache = {}
    im_cur = cv2.imread(imname, -1)
    num_slice = num_slice
    ims = _load_multi_data(im_cur, imname, num_slice)
    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    if window:
        im = im.astype(np.float32, copy=False) - 32768
        im = windowing(im, window)
    #im = windowing(im, [-1024, 1050])
    return im

def windowing(im, win):
    # Scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

