import numpy as np
import tensorflow as tf
import cv2

##----- Read TFRecord-files -----##

def decode_sceneflow(serialized_example, kitty=False):
    with tf.name_scope("decode"):
        features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            "height": tf.FixedLenFeature([], tf.int64),
            "width": tf.FixedLenFeature([], tf.int64),
            "depth": tf.FixedLenFeature([], tf.int64),
            "image_raw_left": tf.FixedLenFeature([], tf.string),
            "image_raw_right": tf.FixedLenFeature([], tf.string),
            "label_left": tf.FixedLenFeature([], tf.string),
            "label_right": tf.FixedLenFeature([], tf.string)
        })

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        height = tf.cast(features["height"], tf.int32)
        width = tf.cast(features["width"], tf.int32)
        depth = tf.cast(features["depth"], tf.int32)

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image_left = tf.decode_raw(features["image_raw_left"], tf.uint8)
        image_right = tf.decode_raw(features["image_raw_right"], tf.uint8)
        disp_left = tf.decode_raw(features["label_left"], tf.float32)
        disp_right = tf.decode_raw(features["label_right"], tf.float32)

        # Set shapes & add batch dimensions
        image_left = tf.cast(tf.reshape(image_left, [540, 960, 3]), dtype=tf.float32)
        image_right = tf.cast(tf.reshape(image_right, [540, 960, 3]), dtype=tf.float32)
        disp_left = tf.reshape(disp_left, [540, 960, 1])
        disp_right = tf.reshape(disp_right, [540, 960, 1])


        images = tf.stack([image_left, image_right])
        labels = tf.stack([disp_left, disp_right])

    return images, labels

def decode_kitty(serialized_example):
    with tf.name_scope("decode"):
        features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            "height": tf.FixedLenFeature([], tf.int64),
            "width": tf.FixedLenFeature([], tf.int64),
            "depth": tf.FixedLenFeature([], tf.int64),
            "image_raw_left": tf.FixedLenFeature([], tf.string),
            "image_raw_right": tf.FixedLenFeature([], tf.string),
            "label_left": tf.FixedLenFeature([], tf.string),
        })

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        height = tf.cast(features["height"], tf.int32)
        width = tf.cast(features["width"], tf.int32)
        depth = tf.cast(features["depth"], tf.int32)

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image_left = tf.decode_raw(features["image_raw_left"], tf.uint8)
        image_right = tf.decode_raw(features["image_raw_right"], tf.uint8)
        disp_left = tf.decode_raw(features["label_left"], tf.float32)

        # Set shapes & add batch dimensions
        image_left = tf.cast(tf.reshape(image_left, [388, 1240, 3]), dtype=tf.float32)
        image_right = tf.cast(tf.reshape(image_right, [388, 1240, 3]), dtype=tf.float32)
        disp_left = tf.reshape(disp_left, [388, 1240, 1])

        images = tf.stack([image_left, image_right])
        labels = tf.stack([disp_left])
        #props = [height, width, depth]

    return images, labels#, props

def augment(image, label):
    with tf.name_scope("augment"):
        crop_height = 256
        crop_width = 512

        seed = np.random.randint(low=1, high=32000, dtype=np.int16)
        image_cropped = tf.random_crop(image, size=[2, crop_height, crop_width, 3], seed=seed)
        label_cropped = tf.random_crop(label, size=[2, crop_height, crop_width, 1], seed=seed)
    return image_cropped, label_cropped

def norm_img(img, label):
    with tf.name_scope("norm_img"):
        """ Normalize input values of images (range [0, max]) to range [r_min, r_max] along dim. """
        img = tf.cast(img, dtype=tf.float32)
        r_min = -1
        r_max = 1
        dim = "HWC"

        # Is batch-dimension the first or last one?
        if dim == "NHWC":
            touple = [1, 2, 3]
        else:
            touple = [0, 1, 2]

        #img = tf.cast(img, tf.float32)
        min_val_left = tf.reduce_min(img[0], axis=touple, keepdims=True)
        min_val_right = tf.reduce_min(img[1], axis=touple, keepdims=True)
        max_val_left = tf.reduce_max(img[0], axis=touple, keepdims=True)
        max_val_right = tf.reduce_max(img[1], axis=touple, keepdims=True)
        # If min not 0 => subtract each image with it's minimum value to get range [0, x]
        img_l = img[0] - min_val_left
        img_r = img[1] - min_val_right
        #min_val = np.squeeze(min_val)
        max_val_left_u = max_val_left - min_val_left
        max_val_right_u = max_val_right - min_val_right

        # Norm
