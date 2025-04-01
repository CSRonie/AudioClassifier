import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import pandas as pd


# check available devices for mic input
# print(sd.query_devices())


# Skipped data gathering, preprocessing, and model training due to time constraints and for faster development,
# we use yamnet model from tensorflow hub for audio classification instead.
 
# references:
# https://www.kaggle.com/models/google/yamnet/code
# https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
# https://www.tensorflow.org/hub/tutorials/yamnet
# https://research.google.com/audioset/dataset/index.html

# YAMNet model 
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# YAMNet class map
class_map_path = "yamnet_class_map.csv"
class_map = pd.read_csv(class_map_path)
class_names = class_map['display_name'].tolist()

# color mapping for actions
# the value can be change into pin no for GPIO output
# or any other action you want to perform based on the sound detected. 

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

# Filter only needed action mapping for optimization and faster prediction
filtered_classes = [name for name in class_names if name in action_mapping]
filtered_indices = [class_names.index(name) for name in filtered_classes]


def record_audio(duration=1, sr=16000,device=3):
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32', device=device)
    sd.wait()
    return np.squeeze(audio)

def classify_audio(audio_data):
    waveform = np.array(audio_data, dtype=np.float32).reshape(-1)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    filtered_scores = scores.numpy().mean(axis=0)[filtered_indices]

    # TOP Prediction 
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
            
            # Map detected sound to an action if available
            # this can be Change to GPIO pin or other action as needed like LED on/off
            action = action_mapping.get(predicted_label, "No action defined")
            print(f"Action: {action}")
    
    except KeyboardInterrupt:
        print("\nStopping listening.")

if __name__ == "__main__":#
    main()