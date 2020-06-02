# %% [markdown]
# ## Record Audio

# %% [code]
import wave
import keras
import pyaudio
import pandas as pd
import numpy as np

import IPython.display as ipd

# %% [code]
# =============================================================================
# CHUNK = 1024 
# FORMAT = pyaudio.paInt16 #paInt8
# CHANNELS = 2 
# RATE = 44100 #sample rate
# RECORD_SECONDS = 3
# WAVE_OUTPUT_FILENAME = "demo_audio.wav"
emotions=["Anger","disgust","fear","happy","Neutral", "sad", "surprise"]
# 
# p = pyaudio.PyAudio()
# 
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK) #buffer
# 
# print("* recording")
# 
# frames = []
# 
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data) # 2 bytes(16 bits) per channel
# 
# print("* done recording")
# 
# stream.stop_stream()
# stream.close()
# p.terminate()
# 
# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()
# =============================================================================

# %% [markdown]
# ## Load The SER Model

# %% [code] {"scrolled":true}
# loading json and creating model
from keras.models import model_from_json
json_file = open('./utils/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./Trained_Models/Speech_Emotion_Recognition_Model.h5")
#print("Loaded model from disk")

# %% [code]
demo_audio_path = './demo_audio.wav'
ipd.Audio(demo_audio_path)

# %% [code]
from utils.feature_extraction import get_audio_features
demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(demo_audio_path, 20000)

# %% [code]
mfcc = pd.Series(demo_mfcc)
pit = pd.Series(demo_pitch)
mag = pd.Series(demo_mag)
C = pd.Series(demo_chrom)
demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)

# %% [code]
demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
demo_audio_features= np.expand_dims(demo_audio_features, axis=2)

# %% [code]
#print(demo_audio_features.shape)

# %% [code]
livepreds = loaded_model.predict(demo_audio_features, 
                         batch_size=32, 
                         verbose=1)

# %% [code]
#print(livepreds)

# %% [code]
#emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
index = livepreds.argmax(axis=1).item()


# %% [code]
print(emotions[index])