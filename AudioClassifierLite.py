import numpy as np
import sounddevice as sd
import tensorflow.lite as tflite
import pandas as pd

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="yamnet.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load YAMNet class map
class_map_path = "yamnet_class_map.csv"
class_map = pd.read_csv(class_map_path)
class_names = class_map['display_name'].tolist()

# Define action mappings
action_mapping = {
    "Gunshot, gunfire": "red",
    "Dog": "red",
    "Bark": "red",
    "Shatter": "red",
    "Air horn, truck horn": "red",

    "Ding-dong": "green",
    "Sliding door": "green",
    "Knock": "green",
    "Tap": "green",

    "Siren": "orange",
    "Slam": "orange",
    "Civil defense siren": "orange",
    "Honk": "orange",

    "Emergency vehicle": "orange",
    "Police car (siren)": "orange",
    "Ambulance (siren)": "orange",
    "Fire engine, fire truck (siren)": "orange",

    "Silence": "white",
    "Noise": "white",
    "Static": "white"
}

# Filter only needed action mappings
filtered_classes = [name for name in class_names if name in action_mapping]
filtered_indices = [class_names.index(name) for name in filtered_classes]

def record_audio(duration=1, sr=16000, device=3):
    """Record audio from microphone"""
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32', device=device)
    sd.wait()
    return np.squeeze(audio)

def classify_audio(audio_data):
    """Run inference on audio data"""
    audio_data = np.expand_dims(audio_data, axis=0)
    waveform = np.array(audio_data, dtype=np.float32).reshape(-1, 1)
    
    # Set the tensor to the audio data
    interpreter.set_tensor(input_details[0]['index'], waveform)
    interpreter.invoke()

    # Get the output prediction
    scores = interpreter.get_tensor(output_details[0]['index'])

    # Filter relevant scores
    filtered_scores = np.mean(scores, axis=0)[filtered_indices]

    # Get top predicted class
    top_index = np.argmax(filtered_scores)
    if top_index >= len(filtered_classes):
        print(f"Warning: Index {top_index} is out of range for filtered class names.")
        return "Unknown"

    return filtered_classes[top_index]

def main():
    print("Listening for sounds. Press Ctrl+C to stop.")

    try:
        while True:
            audio_data = record_audio()
            predicted_label = classify_audio(audio_data)
            print(f"Detected sound: {predicted_label}")

            # Map detected sound to an action
            action = action_mapping.get(predicted_label, "No action defined")
            print(f"Action: {action}")

    except KeyboardInterrupt:
        print("\nStopping listening.")

if __name__ == "__main__":
    main()
