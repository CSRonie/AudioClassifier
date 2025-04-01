import tensorflow as tf

yamnet_model_path = "yamnet_saved_model"

converter = tf.lite.TFLiteConverter.from_saved_model(yamnet_model_path)
tflite_model = converter.convert()

with open("yamnet.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as yamnet.tflite")
