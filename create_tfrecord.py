# import os
# import numpy as np
# import tensorflow as tf
# import math
# import shutil
# import time
# from matplotlib import pyplot as plt
# import cv2


# def main():
    # # path to output TFRecord
    # # tfrecord_path  = './tfrecords/demo/demo.record'
    # tfrecord_path  = './tfrecords/archive/archive.record'
    
    # # path to input file lists
    # src_txt        = './dataset/webface.txt'
    # tar_txt        = None 
    # # shuffle before creating TFRecords
    # shuffle        = True
    # # set random seed
    # seed = 1234
    # np.random.seed(seed)
    # # number of elements in each shard
    # split_size     = 1000
    # # default weight for weighted loss
    # weight_default = 1.0


    # # mkdir
    # tfrecord_dir = os.path.dirname(tfrecord_path)
    # if not os.path.exists(os.path.dirname(tfrecord_path)):
        # os.makedirs(tfrecord_dir)

    # # backup current script
    # #fname, ext = os.path.splitext(os.path.basename(__file__))
    # #now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    # #shutil.copy(__file__, os.path.join(tfrecord_dir, fname + '_' + now + ext))

    # # compute weight for weighted loss
    # weight = weight_default
    # annotations_src = load_annotations(src_txt)

    # dataset = {}
    # # src
    # for annotation in annotations_src:
        # line = annotation.split()
        # image_path = line[0]
        # label = np.int(line[1])
        # dataset[image_path] = [label]

    # # shuffle
    # if shuffle:
        # _ = list(dataset.items())
        # np.random.shuffle(_)
        # dataset = dict(_)

    # num_images = len(dataset)
    # print('Creating TFRecords from {} images'.format(num_images))
    # counter = 0
    # with tf.io.TFRecordWriter(tfrecord_path) as record_writer:
        # for image_path, [label] in dataset.items():
            # image = tf.io.gfile.GFile(image_path, 'rb').read()
            # example = tf.train.Example(features = tf.train.Features(
                # feature={
                    # "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    # "label"  :tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                # }
            # ))

            # record_writer.write(example.SerializeToString())
            # counter += 1
            # if counter % 100 == 0:
                # print('progress {}/{}'.format(counter, num_images))

    # print('Saving {} images in {}'.format(counter, tfrecord_path))

    # # split tfrecord into shards
    # if split_size > num_images:
        # split_size = num_images
    # num_shards = math.ceil(num_images / split_size)
    # with tf.Graph().as_default(), tf.Session() as sess:
        # ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        # batch = ds.make_one_shot_iterator().get_next()
        # part_num = 0
        # while True:
            # try:
                # records = sess.run(batch)
                # part_path = tfrecord_path + '-{:05d}-of-{:05d}'.format(part_num, num_shards)
                # with tf.io.TFRecordWriter(part_path) as writer:
                    # for record in records:
                        # writer.write(record)
                # part_num += 1
            # except tf.errors.OutOfRangeError: break

    # # remove unnecessary file
    # os.remove(tfrecord_path)


# def load_annotations(path):
    # with open(path, 'r') as f:
        # annotations = [line.strip() for line in f.readlines()]
    # return annotations


# if __name__ == '__main__':
    # main()

import os
import numpy as np
import tensorflow as tf
import math
import shutil
import time
import cv2


def main():
    # path to output TFRecord
    tfrecord_path  = './tfrecords/archive/archive.record'
    # path to input file lists
    src_txt        = './dataset/webface.txt'
    tar_txt        = None 
    # shuffle before creating TFRecords
    shuffle        = True
    # set random seed
    seed = 1234
    np.random.seed(seed)
    # number of elements in each shard
    split_size     = 1000
    # default weight for weighted loss
    weight_default = 1.0


    # mkdir
    tfrecord_dir = os.path.dirname(tfrecord_path)
    if not os.path.exists(os.path.dirname(tfrecord_path)):
        os.makedirs(tfrecord_dir)

    # backup current script
    #fname, ext = os.path.splitext(os.path.basename(__file__))
    #now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    #shutil.copy(__file__, os.path.join(tfrecord_dir, fname + '_' + now + ext))

    # compute weight for weighted loss
    weight = weight_default
    annotations_src = load_annotations(src_txt)

    dataset = {}
    # src
    for annotation in annotations_src:
        line = annotation.split()
        image_path = line[0]
        label = np.int(line[1])
        dataset[image_path] = [label]

    # shuffle
    if shuffle:
        _ = list(dataset.items())
        np.random.shuffle(_)
        dataset = dict(_)

    num_images = len(dataset)
    print('Creating TFRecords from {} images'.format(num_images))
    counter = 0
    with tf.io.TFRecordWriter(tfrecord_path) as record_writer:
        for image_path, [label] in dataset.items():
            # 读取图片并转换为灰度图
            img = cv2.imread(image_path)
            if img is None:
                print(f'Warning: Failed to read image {image_path}, skipping...')
                continue
            
            # 转换为灰度图 (BGR -> Grayscale)
            # 如果图片是三通道，转换为灰度；如果已经是灰度，保持不变
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 2:
                img_gray = img
            else:
                print(f'Warning: Unexpected image shape {img.shape} for {image_path}, skipping...')
                continue
            
            # 将灰度图编码为JPEG格式（单通道，shape为 (112, 112)）
            # cv2.imencode 可以处理2D数组（单通道灰度图）
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, image_bytes = cv2.imencode('.jpg', img_gray, encode_param)
            image = image_bytes.tobytes()
            
            example = tf.train.Example(features = tf.train.Features(
                feature={
                    "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    "label"  :tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                }
            ))

            record_writer.write(example.SerializeToString())
            counter += 1
            if counter % 100 == 0:
                print('progress {}/{}'.format(counter, num_images))

    print('Saving {} images in {}'.format(counter, tfrecord_path))

    # split tfrecord into shards
    if split_size > num_images:
        split_size = num_images
    num_shards = math.ceil(num_images / split_size)
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()
        part_num = 0
        while True:
            try:
                records = sess.run(batch)
                part_path = tfrecord_path + '-{:05d}-of-{:05d}'.format(part_num, num_shards)
                with tf.io.TFRecordWriter(part_path) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError: break

    # remove unnecessary file
    os.remove(tfrecord_path)


def load_annotations(path):
    with open(path, 'r') as f:
        annotations = [line.strip() for line in f.readlines()]
    return annotations


if __name__ == '__main__':
    main()
