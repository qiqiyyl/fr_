import tensorflow as tf
import numpy as np
import cv2
import random

random.seed(0)

# need switch to tensorflow2.6 to convert int8 model
frozen_weight_graph = "./checkpoint/demo/freeze.pb"
output_path = "./checkpoint/demo/freeze.tflite"
representative_file_path = './dataset/demo.txt'

input_h = 112
input_w = 112
branch = 1
input_node = 'input'
output_node = 'MobileFaceNet/Logits/LinearConv1x1/BatchNorm/FusedBatchNormV3'

quant_aware = False
inference_input_type = tf.int8
inference_output_type = tf.int8

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(frozen_weight_graph, [input_node], [output_node])


def image_pad_resize(image, target_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:, :, np.newaxis]

    ih, iw = target_size
    h,  w, ch = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    image_resized = image_resized[:, :, np.newaxis]

    image_paded = np.full(shape=[ih, iw, 1], fill_value=128)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    #image_paded = image_paded / 255.

    image_paded = np.array(image_paded, dtype=np.uint8)

    return image_paded


def representative_dataset_gen_yolo():
    file_path = representative_file_path
    with open(file_path, 'r') as f:
        filename = f.readlines()
    random.shuffle(filename)
    if len(filename) > 1000:
        filename = filename[:1000]
    count = 1
    for file in filename:
        print(file)
        file = file.split(' ')[0]
        image = cv2.imread(file)
        image_run = image_pad_resize(image, [input_h, input_w])

        array = np.array(image_run)
        # array = np.expand_dims(array, axis=2)
        array = np.expand_dims(array, axis=0)
        array = array.astype(np.float32)
        array = (array - 128) * 0.0078125
        yield ([array])
        # if count >0:
        #     break


if __name__ == '__main__':
    if quant_aware:
        '''quantization aware-training converter'''
        converter.inference_type = tf.uint8
        input_arrays = converter.get_input_arrays()
        converter.quantized_input_stats = {input_arrays[0]: (127.5, 127.5)}
    else:
        '''post-training converter'''
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen_yolo
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    open(output_path, 'wb').write(tflite_quant_model)
    print('Finished TF-Lite conversion!!!!')