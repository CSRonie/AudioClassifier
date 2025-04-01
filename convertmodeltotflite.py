import tensorflow as tf
import tensorflow_hub as hub

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
def infer(waveform):
    return yamnet_model(waveform)

yamnet_model_path = "yamnet_saved_model"
tf.saved_model.save(
    yamnet_model,
    yamnet_model_path,
    signatures={"serving_default": infer}
)

print(f"Model saved with signature to: {yamnet_model_path}")
