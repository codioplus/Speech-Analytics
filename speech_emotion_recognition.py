
import os

# LibROSA provides the audio analysis
import librosa
# Need to implictly import from librosa
import librosa.display

# Import the audio playback widget
import IPython.display as ipd
from IPython.display import Image

# Enable plot in the notebook
# % pylab inline
# %matplotlib inline
import matplotlib.pyplot as plt

# These are generally useful to have around
import numpy as np
import pandas as pd


# To build Neural Network and Create desired Model
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D #, AveragePooling1D
from keras.layers import Flatten, Dropout, Activation # Input, 
from keras.layers import Dense #, Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

"""# 3. Data Preparation

### Plotting the audio file's waveform and its spectrogram
"""

data, sampling_rate = librosa.load('../speech-emotion-recognition-ravdess-data/Actor_01/03-01-01-01-01-01-01.wav')
# To play audio this in the jupyter notebook
ipd.Audio('../speech-emotion-recognition-ravdess-data/Actor_01/03-01-01-01-01-01-01.wav')

print(len(data))

print(sampling_rate)

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

"""### Setup the Basic Paramter"""

dataset_path = os.path.abspath('./Dataset')
destination_path = os.path.abspath('./')
# To shuffle the dataset instances/records
randomize = True
# for spliting dataset into training and testing dataset
split = 0.8
# Number of sample per second e.g. 16KHz
sampling_rate = 20000 
emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]

"""### Converting Dataset in CSV format

it will cause easy operation on Dataset.
"""

# loading dataframes using dataset module 
from utils import dataset

# To know more about "create_and_load_meta_csv_df" function and it's working, go to "./utils/dataset.py" script. 
df, train_df, test_df = dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize, split)

#print('Dataset samples  : ', len(df),"\nTraining Samples : ", len(train_df),"\ntesting Samples  : ", len(test_df))

"""# 4. Data Visualization

Let's understand what is our dataset.
"""

df.head()

print("Actual Audio : ", df['path'][0])
print("Labels       : ", df['label'][0])

"""### Labels Assigned for emotions : 
- 0 : anger
- 1 : disgust
- 2 : fear
- 3 : happy
- 4 : neutral 
- 5 : sad
- 6 : surprise
"""

unique_labels = train_df.label.unique()
unique_labels.sort()
print("unique labels in Emtion dataset : ")
print(*unique_labels, sep=', ')
unique_labels_counts = train_df.label.value_counts(sort=False)
print("\n\nCount of unique labels in Emtion dataset : ")
print(*unique_labels_counts,sep=', ')

# Histogram of the classes
plt.bar(unique_labels, unique_labels_counts,align = 'center', width=0.6, color = 'c')
plt.xlabel('Number of labels', fontsize=16)
plt.xticks(unique_labels)
plt.ylabel('Count of each labels', fontsize=16)
plt.title('Histogram of the Labels', fontsize=16)
plt.show()

"""# 5. Data Pre-Processing

### Getting the features of audio files using librosa

Calculating MFCC, Pitch, magnitude, Chroma features.
"""

Image('./images/feature_plots.png')

from utils.feature_extraction import get_features_dataframe
from utils.feature_extraction import get_audio_features

"""To know more about "get_features_dataframe" and "get_audio_features" functions and their working, go to "./utils/feature_extraction.py" script."""

# trainfeatures, trainlabel = get_features_dataframe(train_df, sampling_rate)
# testfeatures, testlabel = get_features_dataframe(test_df, sampling_rate)

# I have ran above 2 lines and get the featured dataframe. 
# and store it into pickle file to use it for later purpose.
# it take too much time to generate features(around 30-40 minutes).

trainfeatures = pd.read_pickle('./features_dataframe/trainfeatures')
trainlabel = pd.read_pickle('./features_dataframe/trainlabel')
testfeatures = pd.read_pickle('./features_dataframe/testfeatures')
testlabel = pd.read_pickle('./features_dataframe/testlabel')

trainfeatures.shape

trainfeatures = trainfeatures.fillna(0)
testfeatures = testfeatures.fillna(0)

# By using .ravel() : Converting 2D to 1D e.g. (512,1) -> (512,). To prevent DataConversionWarning

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel).ravel()
X_test = np.array(testfeatures)
y_test = np.array(testlabel).ravel()

y_train[:5]

# One-Hot Encoding
lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

y_train[:5]

"""### Changing dimension for CNN model"""

x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)

x_traincnn.shape


opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)




"""### Loading the model"""

# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./Trained_Models/Speech_Emotion_Recognition_Model.h5")
#print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

"""# 8. Test Set Prediction

### Predicting emotions on the test data
"""

preds = loaded_model.predict(x_testcnn, 
                         batch_size=32, 
                         verbose=1)

#print(preds)

preds1=preds.argmax(axis=1)

#print(preds1)

abc = preds1.astype(int).flatten()

predictions = (lb.inverse_transform((abc)))

preddf = pd.DataFrame({'predictedvalues': predictions})
preddf[:10]

actual=y_test.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))

actualdf = pd.DataFrame({'actualvalues': actualvalues})
actualdf[:10]

finaldf = actualdf.join(preddf)

"""## Actual v/s Predicted emotions"""

#print(finaldf[130:140])

finaldf.groupby('actualvalues').count()

finaldf.groupby('predictedvalues').count()

finaldf.to_csv('Predictions.csv', index=False)

"""# 9. Live Demonstration

#### The file 'output10.wav' in the next cell is the file that was recorded live using the code in AudioRecoreder notebook found in the repository
"""

demo_audio_path = './Dataset/anger/anger008.wav'

demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(demo_audio_path,sampling_rate)

mfcc = pd.Series(demo_mfcc)
pit = pd.Series(demo_pitch)
mag = pd.Series(demo_mag)
C = pd.Series(demo_chrom)
demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)

demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
demo_audio_features= np.expand_dims(demo_audio_features, axis=2)

#print(demo_audio_features.shape)

livepreds = loaded_model.predict(demo_audio_features, 
                         batch_size=32, 
                         verbose=1)

#print(livepreds)

# emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
index = livepreds.argmax(axis=1).item()
#print(index)

print(emotions[index])

"""# Thank You !"""