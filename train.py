from glob import glob

from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import *

from sklearn.model_selection import train_test_split
from metrics import *
from model import *
from image import gen

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    model = build_model((640, 640, 3), 2)

    filepath = './model/model.h5'  ##模型权重存放位置
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='loss', verbose=0, save_weights_only=True,
                                   save_best_only=True)
    rlu = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0, mode='auto', cooldown=0, min_lr=0)
    metrics = [
        dice_coef,
        iou,
        Recall(),
        Precision()
    ]

    model.compile(optimizer=Adam(lr=0.0001, clipvalue=0.5), loss='binary_crossentropy', metrics=metrics)

    paths = glob('./train_data/*.json')  ##table line dataset label with labelme
    trainP, testP = train_test_split(paths, test_size=0.1)
    print('total:', len(paths), 'train:', len(trainP), 'test:', len(testP))
    batchsize = 8
    trainloader = gen(trainP, batchsize=batchsize, linetype=1)
    testloader = gen(testP, batchsize=batchsize, linetype=1)
    model.fit_generator(trainloader,
                        steps_per_epoch=max(1, len(trainP) // batchsize),
                        callbacks=[checkpointer],
                        validation_data=testloader,
                        validation_steps=max(1, len(testP) // batchsize),
                        epochs=300)

