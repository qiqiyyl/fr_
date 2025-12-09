import os
import numpy as np
import cv2
import tensorflow as tf


model_path = './checkpoint/demo/freeze.tflite'

img_0_path = './images/test/0.png'
img_1_path = './images/test/1.png'
img_2_path = './images/test/2.png'
img_3_path = './images/test/3.png'


interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
qscale = output_details[0]["quantization"][0]
qoffset = output_details[0]["quantization"][1]

def CosSim(emb_array1, emb_array2):
    dot = sum(a * b for a, b in zip(emb_array1, emb_array2))
    norm_a = sum(a * a for a in emb_array1) ** 0.5
    norm_b = sum(b * b for b in emb_array2) ** 0.5
    sim = dot / (norm_a * norm_b)

    return sim

def feature_extraction(img):
   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f = np.asarray(img, dtype=float)
    img_f = img_f - 128
    image_r_in = np.asarray(img_f, dtype=np.int8)
    image_r_in = np.expand_dims(image_r_in, axis=0)
    image_r_in = np.expand_dims(image_r_in, axis=-1)
    interpreter.set_tensor(input_details[0]['index'], image_r_in)
    interpreter.invoke()
    q_features = interpreter.get_tensor(output_details[0]['index'])
    emb_array = qscale * (q_features.astype(float) - qoffset)
    emb_array = np.squeeze(np.asarray(emb_array))

    return emb_array

def cal_sim(img_0,img_1):

    emb_array_0 = feature_extraction(img_0)
    emb_array_1 = feature_extraction(img_1)
    sim = CosSim(emb_array_0, emb_array_1)
    return sim
    
img_0 = cv2.imread(img_0_path)
img_1 = cv2.imread(img_1_path)
img_2 = cv2.imread(img_2_path)
img_3 = cv2.imread(img_3_path)

sim01 = cal_sim(img_0,img_1)
sim12 = cal_sim(img_1,img_2)
sim23 = cal_sim(img_2,img_3)

print("similarity of (%s,%s): %.4f"%(os.path.basename(img_0_path),os.path.basename(img_1_path),sim01))
print("similarity of (%s,%s): %.4f"%(os.path.basename(img_1_path),os.path.basename(img_2_path),sim12))
print("similarity of (%s,%s): %.4f"%(os.path.basename(img_2_path),os.path.basename(img_3_path),sim23))
