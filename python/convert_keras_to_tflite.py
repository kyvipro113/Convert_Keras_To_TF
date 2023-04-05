from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model


json_model_save = open("Model/imgSeg_brainMRI_unet.json", "r")
load_model_json = json_model_save.read()
json_model_save.close()
model = model_from_json(load_model_json)
model.load_weights("Model/brain_seg_unet.hdf5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("Model_TF/unet_brain_segmentation_edge.tflite", "wb").write(tflite_model)