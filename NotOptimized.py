import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import pandas as pd


#Skipped data gathering, preprocessing, and model training due to time constraints and for faster development,
#we use yamnet model from tensorflow hub for audio classification instead.

# references:
# https://www.kaggle.com/models/google/yamnet/code
# https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
# https://www.tensorflow.org/hub/tutorials/yamnet
# https://research.google.com/audioset/dataset/index.html

#YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

#YAMNet class
class_map_path = "yamnet_class_map.csv"
class_names = pd.read_csv(class_map_path)['display_name'].tolist()

def record_audio(duration=1, sr=16000):
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)


def classify_audio(audio_data, sr=16000):
    waveform = np.array(audio_data, dtype=np.float32).reshape(-1)
    
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    #Top Prediction
    avg_scores = scores.numpy().mean(axis=0)
    top_index = np.argmax(avg_scores)

    if top_index >= len(class_names):
        print(f"Warning: Index {top_index} is out of range for class_names.")
        return "Unknown"

    return class_names[top_index]

# color mapping for actions
action_mapping = {

    "Gunshot, gunfire": "red",
    "Dog": "red",
    "Bark": "red",
    "Shatter": "red",

    "Ding-dong": "green",
    "Sliding green": "green",
    "Knock": "green",
    "Tap": "green",

    "Siren": "orange",
    "Slam": "orange",
    "Civil defense siren": "orange",
    "Honk": "orange",
    "Air horn, truck horn": "orange",
    "Emergency vehicle": "orange",
    "Police car (siren)": "orange",
    "Ambulance (siren)": "orange",
    "Fire engine, fire truck (siren)": "orange",

    "Silence": "white",
    "Noise": "white",
    "Static": "white"
}



def main():
    print("Listening for sounds. Press Ctrl+C to stop.")
    
    try:
        while True:
            audio_data = record_audio()
            predicted_label = classify_audio(audio_data)
            print(f"Detected sound: {predicted_label}")
            
            if predicted_label in action_mapping:
                print(f"Action: {action_mapping[predicted_label]}")
    
    except KeyboardInterrupt:
        print("\nStopping listening.")

if __name__ == "__main__":
    main()
