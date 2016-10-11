import codecs
import json
import os
import proj_constants
import numpy as np
import tensorflow as tf
import threading

IMAGE_SHAPE = 2*proj_constants.WIDTH*proj_constants.HEIGHT
LABEL_SIZE = 70

train_data = json.load(codecs.open(os.path.join(proj_constants.DATA_DIR, 'annotation_train.json'),
                                   'r', encoding='utf-8'))['values']
test_data = json.load(codecs.open(os.path.join(proj_constants.DATA_DIR, 'annotation_test.json'),
                                  'r', encoding='utf-8'))['values']


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def extract_filenames(data):
    filenames = []
    for i in xrange(len(data)):
        filenames.append(data[i]['filename'])
    return filenames


def extract_labels(data):
    labels = []
    for i in xrange(len(data)):
        labels.append(data[i]['predicate_index'])
    return np.asarray(labels)


def data_to_tfrecords(image_filenames, labels, tfrecords_file):
    """Writes images and labels to a file"""
    file_queue = tf.train.string_input_producer(image_filenames)
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    decoded_img = tf.image.decode_jpeg(value)

    init_op = tf.initialize_all_variables()

    writer = tf.python_io.TFRecordWriter(tfrecords_file)

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in xrange(len(image_filenames)):
            image = decoded_img.eval()
            image = image*(1./255.)
            image_raw = image.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'x': _bytes_feature(image_raw),
                'y': _int64_feature(int(labels[i]))
            }))
            writer.write(example.SerializeToString())

        coord.request_stop()
        coord.join(threads)


def write_tfrecords():
    """Converts images to tfrecords. Creates one tfrecords file per training example."""

    # Create separate tfrecords file for each training example
    train_filenames = extract_filenames(train_data)
    train_labels = extract_labels(train_data)
    for i in xrange(len(train_filenames)):
        curr_train_file = os.path.join(proj_constants.DATA_DIR, "train_images", train_filenames[i])
        train_tfrecords_file = os.path.join(proj_constants.DATA_DIR, "train_tfrecords", train_filenames[i]+'.tfrecords')
        curr_train_label = train_labels[i]
        train_thr = threading.Thread(target=data_to_tfrecords, args=([curr_train_file], [curr_train_label], train_tfrecords_file), kwargs={})
        if i % 500 == 0:
            print("Converting %d th training image to tfrecords\n" % i)
        train_thr.start()

    # Create separate tfrecords file for each test example
    test_filenames = extract_filenames(test_data)
    test_labels = extract_labels(test_data)
    for i in xrange(len(test_filenames)):
        curr_test_file = os.path.join(proj_constants.DATA_DIR, "test_images", test_filenames[i])
        test_tfrecords_file = os.path.join(proj_constants.DATA_DIR, "test_tfrecords", test_filenames[i]+'.tfrecords')
        curr_test_label = test_labels[i]
        test_thr = threading.Thread(target=data_to_tfrecords,
                                     args=([curr_test_file], [curr_test_label], test_tfrecords_file), kwargs={})
        if i % 500 == 0:
            print("Converting %d th test image to tfrecords\n" % i)
        test_thr.start()

write_tfrecords()

'''
References:
1. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
2. http://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow
3. http://stackoverflow.com/questions/35106101/why-does-my-tensorflow-convnet-attempted-training-result-in-nan-gradients
'''
