# -*- coding: utf-8 -*-
"""
基于 YOLO3 (https://github.com/qqwweee/keras-yolo3) 实现的物体检测代码
在 ModelArts Notebook 中的代码运行方法：
（1）训练
cd {train.py所在目录}
python train.py --local_data_root='/home/ma-user/work/' --data_url='obs://myobsbucket0408/datasets/' --train_url='obs://myobsbucket0408/train_output/' --max_epochs_1=5 --max_epochs_2=10
"""
import os
import codecs
import shutil
import argparse

try:
    import moxing as mox
except:
    print('not use moxing')
import xml.etree.ElementTree as ET

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

parser = argparse.ArgumentParser(description='YOLO3 Training')
parser.add_argument('--max_epochs_1', default=5, type=int, help='number of total epochs to run in stage 1')
parser.add_argument('--max_epochs_2', default=10, type=int, help='number of total epochs to run in total')
parser.add_argument('--initial_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size, default 32')
parser.add_argument('--score', default=0.3, type=float, help='score, default 0.3')
parser.add_argument('--iou', default=0.45, type=float, help='iou, default 0.45')
parser.add_argument('--local_data_root', default='/cache/', type=str,
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
parser.add_argument('--train_local', default='', type=str, help='the training output results on local')


def _main(args):
    annotation_path = args.txt_anno_path
    log_dir = args.train_local
    classes_path = os.path.join(os.path.dirname(__file__), 'model_data/train_classes.txt')
    anchors_path = os.path.join(os.path.dirname(__file__), 'model_data/train_anchors.txt')
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path=os.path.join(os.path.dirname(__file__),
                                                                      '../pre-trained_weights/yolo_weights.h5'))  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = args.batch_size
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=args.max_epochs_1,
                            initial_epoch=args.initial_epoch,
                            callbacks=[logging, checkpoint])
        model.save_weights(os.path.join(log_dir, 'trained_weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = args.batch_size  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=args.max_epochs_2,
                            initial_epoch=args.max_epochs_1,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(os.path.join(log_dir, 'trained_weights_final.h5'))

    # Further training if needed.
    gen_model_dir(log_dir, args, classes_path, anchors_path)


def gen_model_dir(log_dir, args, classes_path, anchors_path):
    if args.train_url.startswith('s3://') or args.train_url.startswith('obs://'):
        mox.file.copy_parallel(log_dir, args.train_url)

    current_dir = os.path.dirname(__file__)
    mox.file.copy(os.path.join(current_dir, 'deploy_scripts/config.json'),
                  os.path.join(args.train_url, 'model/config.json'))  # mox.file.copy可同时兼容本地和OBS路径的拷贝操作
    mox.file.copy(os.path.join(current_dir, 'deploy_scripts/customize_service.py'),
                  os.path.join(args.train_url, 'model/customize_service.py'))
    mox.file.copy_parallel(os.path.join(current_dir, 'yolo3'),
                           os.path.join(args.train_url, 'model/yolo3'))  # 拷贝一个目录得用copy_parallel接口
    mox.file.copy(os.path.join(log_dir, 'trained_weights_final.h5'),
                  os.path.join(args.train_url, 'model/trained_weights_final.h5'))  # 默认拷贝最后一个模型到model目录
    mox.file.copy(classes_path,
                  os.path.join(args.train_url, 'model', os.path.basename(classes_path)))
    mox.file.copy(anchors_path,
                  os.path.join(args.train_url, 'model', os.path.basename(anchors_path)))
    mox.file.copy(os.path.join(current_dir, 'model_data/classify_rule.json'),
                  os.path.join(args.train_url, 'model/classify_rule.json'))

    print('gen_model_dir success, model dir is at', os.path.join(args.train_url, 'model'))


def get_classes(classes_path):
    '''loads the classes'''
    with codecs.open(classes_path, 'r', 'utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def prepare_data_on_modelarts(args):
    """
    将OBS上的数据拷贝到ModelArts中
    """
    # 拷贝预训练参数文件
    src_h5_path = 's3://ma-competitions-bj4/model_zoo/tensorflow/yolo3/yolo_weights.h5'
    dst_h5_path = os.path.join(os.path.dirname(__file__), '../pre-trained_weights/yolo_weights.h5')
    if not os.path.exists(dst_h5_path):
        mox.file.copy(src_h5_path, dst_h5_path)
        print('copy %s to %s success' % (src_h5_path, dst_h5_path))
    else:
        print(dst_h5_path, 'already exists')

    # 默认使用ModelArts中的如下两个路径用于存储数据：
    # 1）/cache/datasets: 存储从OBS拷贝过来的训练数据
    # 2）/cache/log: 存储训练日志和训练模型，并且在训练结束后，该目录下的内容会被全部拷贝到OBS
    if not (args.data_url.startswith('s3://') or args.data_url.startswith('obs://')):
        args.data_local = args.data_url
    else:
        args.data_local = os.path.join(args.local_data_root, 'datasets/trainval')
        if not os.path.exists(args.data_local):
            data_dir = os.path.join(args.local_data_root, 'datasets')
            mox.file.copy_parallel(args.data_url, data_dir)
            os.system('cd %s;unzip trainval.zip' % data_dir)  # 训练集已提前打包为trainval.zip
            if os.path.isdir(args.data_local):
                os.system('cd %s;rm trainval.zip' % data_dir)
                print('unzip trainval.zip success, args.data_local is', args.data_local)
            else:
                raise Exception('unzip trainval.zip Failed')
        else:
            print('args.data_local: %s is already exist, skip copy' % args.data_local)

    shutil.copy(os.path.join(args.data_local, 'train_classes.txt'),
                os.path.join(os.path.dirname(__file__), 'model_data/train_classes.txt'))
    shutil.copy(os.path.join(args.data_local, 'classify_rule.json'),
                os.path.join(os.path.dirname(__file__), 'model_data/classify_rule.json'))

    if not (args.train_url.startswith('s3://') or args.train_url.startswith('obs://')):
        args.train_local = args.train_url
    else:
        args.train_local = os.path.join(args.local_data_root, 'log/')
        if not os.path.exists(args.train_local):
            os.mkdir(args.train_local)

    return args


def convert_voc_anno_to_txt(args):
    """
    将VOC格式的标注转换成txt格式的标注
    """
    root_dir = os.path.join(os.path.abspath(args.data_local), 'VOC2007')
    jpg_dir = os.path.join(root_dir, 'JPEGImages')
    xml_dir = os.path.join(root_dir, 'Annotations')
    trainval_path = os.path.join(root_dir, 'ImageSets/Main/trainval.txt')
    txt_anno_path = os.path.join(root_dir, 'txt_anno.txt')

    with codecs.open(os.path.join(os.path.dirname(__file__), 'model_data/train_classes.txt'), 'r', 'utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    txt_annos = []
    with open(trainval_path) as f:
        img_names = f.readlines()
    for img_name in img_names:
        img_name = img_name.strip()
        jpg_path = os.path.join(jpg_dir, img_name + '.jpg')
        xml_path = os.path.join(xml_dir, img_name + '.xml')

        line = jpg_path
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in class_names:
                continue
            cls_id = class_names.index(cls)
            xmlbox = obj.find('bndbox')
            box = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                   int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            line = line + " " + ",".join([str(v) for v in box]) + ',' + str(cls_id)
        line = line + '\n'
        txt_annos.append(line)
    with open(txt_anno_path, 'w') as f:
        f.writelines(txt_annos)
    args.txt_anno_path = txt_anno_path
    return args


def gen_anchors(args):
    from kmeans import YOLO_Kmeans
    cluster_number = 9
    kmeans = YOLO_Kmeans(cluster_number, args.txt_anno_path)
    kmeans.txt2clusters()


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    args = prepare_data_on_modelarts(args)  # TODO，重要，将OBS上的数据拷贝到 ModelArts 中
    args = convert_voc_anno_to_txt(args)
    gen_anchors(args)
    _main(args)
