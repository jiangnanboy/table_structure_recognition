#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model import build_model
from utils import letterbox_image, get_table_line, adjust_lines, line_to_line

import numpy as np
import cv2
import math
import time
import os
from utils import draw_lines
from PIL import Image

model = build_model((640, 640, 3), 2)
model.load_weights('model/model.h5')

def table_line(img, size=(640, 640), hprob=0.5, vprob=0.5, row=50, col=30, alph=15):
    sizew, sizeh = size
    inputBlob, fx, fy = letterbox_image(img[..., ::-1], (sizew, sizeh))
    pred = model.predict(np.array([np.array(inputBlob) / 255.0]))
    pred = pred[0]
    vpred = pred[..., 1] > vprob  ##竖线
    hpred = pred[..., 0] > hprob  ##横线
    vpred = vpred.astype(int)
    hpred = hpred.astype(int)
    colboxes = get_table_line(vpred, axis=1, lineW=col)
    rowboxes = get_table_line(hpred, axis=0, lineW=row)
    ccolbox = []
    crowlbox = []
    if len(rowboxes) > 0:
        rowboxes = np.array(rowboxes)
        rowboxes[:, [0, 2]] = rowboxes[:, [0, 2]] / fx
        rowboxes[:, [1, 3]] = rowboxes[:, [1, 3]] / fy
        xmin = rowboxes[:, [0, 2]].min()
        xmax = rowboxes[:, [0, 2]].max()
        ymin = rowboxes[:, [1, 3]].min()
        ymax = rowboxes[:, [1, 3]].max()
        ccolbox = [[xmin, ymin, xmin, ymax], [xmax, ymin, xmax, ymax]]
        rowboxes = rowboxes.tolist()

    if len(colboxes) > 0:
        colboxes = np.array(colboxes)
        colboxes[:, [0, 2]] = colboxes[:, [0, 2]] / fx
        colboxes[:, [1, 3]] = colboxes[:, [1, 3]] / fy

        xmin = colboxes[:, [0, 2]].min()
        xmax = colboxes[:, [0, 2]].max()
        ymin = colboxes[:, [1, 3]].min()
        ymax = colboxes[:, [1, 3]].max()
        colboxes = colboxes.tolist()
        crowlbox = [[xmin, ymin, xmax, ymin], [xmin, ymax, xmax, ymax]]

    rowboxes += crowlbox
    colboxes += ccolbox

    rboxes_row_, rboxes_col_ = adjust_lines(rowboxes, colboxes, alph=alph)
    rowboxes += rboxes_row_
    colboxes += rboxes_col_
    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
        for j in range(ncol):
            rowboxes[i] = line_to_line(rowboxes[i], colboxes[j], 10)
            colboxes[j] = line_to_line(colboxes[j], rowboxes[i], 10)

    return rowboxes, colboxes

def image_resize(img, width, height, max_side_len):
    resize_width = width
    resize_height = height
    ratio = 1.0
    if max(resize_height, resize_width) > max_side_len:
        if resize_height > resize_width:
            ratio = max_side_len / resize_height
        else:
            ratio = max_side_len / resize_width
    resize_height = int(resize_height * ratio)
    resize_width = int(resize_width * ratio)
    if resize_height % 32 == 0:
        pass
    elif math.floor(resize_height / 32) <= 1:
        resize_height = 32
    else:
        resize_height = int(math.floor(resize_height / 32)) * 32
    if resize_width % 32 == 0:
        pass
    elif math.floor(resize_width / 32) <= 1:
        resize_width = 32
    else:
        resize_width = int(math.floor(resize_width / 32)) * 32
    img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    return img

if __name__ == '__main__':

    p = 'img/8.jpg'
    img = cv2.imread(p)
    height, width = img.shape[:2]
    img = image_resize(img, width, height, 1024)
    t = time.time()
    rowboxes, colboxes = table_line(img[..., ::-1], size=(640, 640), hprob=0.5, vprob=0.5)
    img = draw_lines(img, rowboxes + colboxes, color=(255, 0, 0), lineW=2)

    print(time.time() - t, len(rowboxes), len(colboxes))
    cv2.imwrite(os.path.join('img', 'table-line.png'), img)
