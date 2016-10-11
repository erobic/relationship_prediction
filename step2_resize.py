import json
from PIL import Image, ImageDraw
import codecs
import proj_constants
import os


def find_resize_ratio(width, height):
    """Finds the ratio by which the image dimensions should be resized.
    Objective is to have a limit in width and height and preserve the original aspect ratio"""
    width_ratio = proj_constants.WIDTH * 1. / width
    height_ratio = proj_constants.HEIGHT * 1. / height
    resize_ratio = min(width_ratio, height_ratio)
    return resize_ratio


def get_resized_dim(width, height, resize_ratio):
    """Multiplies width and height by the resize_ratio to get new width/height for image"""
    new_width = width * resize_ratio
    new_height = height * resize_ratio
    return int(new_width), int(new_height)


def resize(filename, data):
    """Resizes image and bounding boxes
    Returns:
         resized_im: The resized image
         sub_min: (x_min, y_min) for subject bounding box
         sub_max: (x_max, y_max) for subject bounding box
         obj_min: (x_min, y_min) for object bounding box
         obj_max: (x_max, y_max) for object bounding box"""
    im = Image.open(filename)
    resize_ratio = find_resize_ratio(im.size[0], im.size[1])

    # Find resized dimensions for the bounding box for object
    sub_box = data['subBox'][0]
    sub_min = get_resized_dim(sub_box[2], sub_box[0], resize_ratio)  # params: (x_min, y_min, resize_ratio)
    sub_max = get_resized_dim(sub_box[3], sub_box[1], resize_ratio)  # params: (x_max, y_max, resize_ratio)

    # Find resized dimensions for the bounding box for subject
    obj_box = data['objBox'][0]
    obj_min = get_resized_dim(obj_box[2], obj_box[0], resize_ratio)  # params: (x_min, y_min, resize_ratio)
    obj_max = get_resized_dim(obj_box[3], obj_box[1], resize_ratio)  # params: (x_max, y_max, resize_ratio)

    # Find resized dimensions for image
    im_resized_dim = get_resized_dim(im.size[0], im.size[1], resize_ratio)
    resized_im = im.resize(im_resized_dim, Image.ANTIALIAS)
    return [resized_im, sub_min, sub_max, obj_min, obj_max]


def preprocess(img_info, img_file, new_file):
    """Creates a new image with left half of the image being the subject and right half of the image being the object.
    The pixes outside the bounding boxes are made white.

    Args:
        img_info: Information about image as JSON object. Should have bounding boxes for subject and object.
        img_file: Image file (jpg)
        new_file: New file to save the image"""
    new_im, sub_min, sub_max, obj_min, obj_max = resize(img_file, img_info)
    final_im = Image.new('RGB', (int(proj_constants.WIDTH * 2), int(proj_constants.HEIGHT)), (255, 255, 255))
    sub_box = (sub_min[0], sub_min[1], sub_max[0], sub_max[1])
    obj_box = (obj_min[0], obj_min[1], obj_max[0], obj_max[1])
    sub = new_im.crop(sub_box)
    obj = new_im.crop(obj_box)
    final_im.paste(sub, sub_box)

    # paste object to the right of subject i.e. subject = left half of image, object = right half of image
    obj_paste_box = (obj_min[0] + int(proj_constants.WIDTH), obj_min[1], obj_max[0] + int(proj_constants.WIDTH), obj_max[1])
    final_im.paste(obj, obj_paste_box)
    final_im.save(new_file, "JPEG", quality=90)

# Resize and save training images
annotation_train_json = json.load(codecs.open(os.path.join(proj_constants.DATA_DIR, 'annotation_train.json'), 'r', encoding='utf-8'))
annotation_train_data = annotation_train_json['values']
for i in xrange(0, len(annotation_train_data)):
    if i % 500 == 0:
        print "Resizing train image#: %d" % i
    old_file = os.path.join(proj_constants.DATA_DIR, 'train_images_orig', annotation_train_data[i]['filename'])
    new_file = os.path.join(proj_constants.DATA_DIR, 'train_images', annotation_train_data[i]['filename'])
    preprocess(annotation_train_data[i], old_file, new_file)

# Resize and save test images
annotation_test_json = json.load(codecs.open(os.path.join(proj_constants.DATA_DIR, 'annotation_test.json'), 'r', encoding='utf-8'))
annotation_test_data = annotation_test_json['values']
for i in xrange(0, len(annotation_test_data)):
    if i % 500 == 0:
        print "Resizing test image#: %d" % i
    old_file = os.path.join(proj_constants.DATA_DIR, 'test_images_orig', annotation_test_data[i]['filename'])
    new_file = os.path.join(proj_constants.DATA_DIR, 'test_images', annotation_test_data[i]['filename'])
    preprocess(annotation_test_data[i], old_file, new_file)