# %% [markdown]
# # TBrain Pathological Voice: Acoustic
# 

# %% [markdown]
# ## Import Package
# 

# %%
path = "tbrain-pathological-voice"

# %%
import sys
sys.path.append(path)

# %%
import os
import time
import random
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

# %%
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

# %%
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

# %% [markdown]
# ## Enabling and testing the TPU
# 

# %%
SEED = 5397
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# %%
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    # Show GPU information
    gpu_info = !nvidia-smi
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        print('Not connected to a GPU')
    else:
        print(gpu_info)
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])

# %% [markdown]
# ## Const & Inputs
# 

# %%
# Number of MFCCs to return
n_mfcc = 20

# Number of folds for cross validation
n_splits = 5

# %% [markdown]
# ## Data Preprocessing
# 

# %% [markdown]
# ### Load Data
# 

# %%
source_df = pd.read_csv(F'{path}/src/data/training_datalist.csv')
source_df['wav_path'] = source_df['ID'].apply(lambda x: F'{path}/src/data/training_voice_data/{x}.wav')
print("source_df.columns :", source_df.columns)

# %% [markdown]
# ### Encode Categorical Features
# 

# %%
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(source_df['Disease category'].values.reshape(-1, 1))
source_df['class'] = list(np.array(enc.transform(source_df['Disease category'].values.reshape(-1, 1)).toarray().tolist()))

# %% [markdown]
# ### Load Voice Features
# 

# %%
def audio_to_mfccs(filename, sample_rate=44100, offset=0, n_mfcc=20, duration=None):
    voice, sample_rate = librosa.load(filename, sr=sample_rate, offset=offset, duration=duration)
    n_fft = int(16/1000 * sample_rate)  # Convert 16 ms to samples
    hop_length = int(8/1000 * sample_rate)  # Convert 8 ms to samples
    mfcc_feature = librosa.feature.mfcc(y=voice, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # mfccs_cvn = (mfcc_feature - np.mean(mfcc_feature, axis=1, keepdims=True)) / np.std(mfcc_feature, axis=1, keepdims=True)
    delta_mfcc_feature = librosa.feature.delta(mfcc_feature)
    mfccs = np.concatenate((mfcc_feature, delta_mfcc_feature))
    mfccs_features = np.transpose(mfccs)  # all frames
    return mfccs_features
    # mfcc_feature.shape : (13, 251)
    # delta_mfcc_feature.shape : (13, 251)
    # mfccs.shape : (26, 251)
    # mfccs_features.shape : (251, 26)

# %% [markdown]
# Read the audio file and convert it to MFCC feature vector
# 

# %%
source_df['mfccs_feature'] = source_df['wav_path'].apply(lambda x: audio_to_mfccs(x, n_mfcc=n_mfcc))
source_df['length'] = source_df['mfccs_feature'].apply(lambda x: x.shape[0])

# %% [markdown]
# Calculate the length of the MFCC feature vector
# 

# %%
group_by_length = source_df.groupby('length').size().reset_index(name='counts')
max_length = group_by_length['length'].max()
group_by_length

# %% [markdown]
# Convert the MFCC feature vector to a fixed length to facilitate the training of the model
# 

# %%
padded_mfcc = np.zeros((max_length, n_mfcc * 2))
source_df['mfccs_feature'] = source_df['mfccs_feature'].apply(lambda x: np.concatenate((x, padded_mfcc[:max_length - x.shape[0]])))
print(np.array(source_df['mfccs_feature'].tolist()).shape)

# %% [markdown]
# Show the MFCC feature vector
# 

# %%
def plot_mfccs_feature(mfccs_feature):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs_feature, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
    
plot_mfccs_feature(source_df['mfccs_feature'][2].T)

# %% [markdown]
# ## Modelling & Training
# 
# Acoustic audio file conversion MFCC features and training
# 

# %%
def time_shift_augment(mfccs, time_shift_rate=0.1, sample_rate=44100):
    # 隨機生成時間偏移量
    time_shift = int(sample_rate * time_shift_rate)
    time_shift_amt = tf.random.uniform(shape=[], minval=-time_shift, maxval=time_shift, dtype=tf.int32)
    # 將 MFCC 特徵向左或向右平移時間軸
    mfccs = tf.roll(mfccs, time_shift_amt, axis=0)
    return mfccs

def pitch_shift(mfccs, pitch_shift=2):
    # 隨機生成音調偏移量
    pitch_shift_amt = tf.random.uniform(shape=[], minval=-pitch_shift, maxval=pitch_shift, dtype=tf.int32)
    # 將 MFCC 特徵向上或向下平移頻率軸
    mfccs = tf.roll(mfccs, pitch_shift_amt, axis=1)
    return mfccs

def noise_augment(mfccs, sigma=0.05):
    # 隨機生成高斯雜訊
    noise = tf.random.normal(shape=tf.shape(mfccs), mean=0.0, stddev=sigma, dtype=tf.float32)
    # 將高斯雜訊加到 MFCC 特徵上
    mfccs = mfccs + noise
    return mfccs

def augment_audio(audio):
    augment_type = np.random.choice(['pitch_shift', 'white_noise', 'None'])
    if augment_type == 'time_shift':
        return time_shift_augment(audio, time_shift_rate=0.05)
    elif augment_type == 'pitch_shift':
        return pitch_shift(audio, pitch_shift=1)
    elif augment_type == 'white_noise':
        return noise_augment(audio, sigma=0.05)
    else:
        return audio

# %%
plot_mfccs_feature(noise_augment(source_df['mfccs_feature'][2]).numpy().T)

# %%
def get_model(input_shape, learning_rate=0.001, dropout=0.5, isAugmentationActivated=False):
    augmentation_layer = tf.keras.layers.Lambda(augment_audio)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    if isAugmentationActivated:
        model.add(augmentation_layer)

    model.add(tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(512, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))

    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(name='recall')])
    return model

# %%
# [10, 30, 50, 100, 150]
EPOCHS = 300

# [16, 32(default), 64, 128, 256, 512, 1024]
BATCH_SIZE = 32 * strategy.num_replicas_in_sync

# Dropout
DROPOUT = 0.3

# learning rate
LR_START = 1e-04

# Define model name
model_name = F'cnnlstm1d_simple_mfcc{n_mfcc}_l{max_length}_augmentation_epochs{EPOCHS}_batchsize{BATCH_SIZE}_lr{LR_START}_dropout{DROPOUT}_reduce_recall'
# model_name = F'cnnlstm1d_mfcc{n_mfcc}_l{max_length}_epochs{EPOCHS}_batchsize{BATCH_SIZE}_lr{LR_START}_dropout{DROPOUT}'

# Check model is exist, stop training if model is exist
if os.path.exists(F'{path}/src/models/{model_name}_fold1.h5'):
    raise SystemExit(F'{model_name} is exist')

# %%
results = pd.DataFrame()
for i in range(0, n_splits):
    train_df = source_df[source_df['foldIdx'].ne(i)].reset_index(drop=True)
    test_df = source_df[source_df['foldIdx'].eq(i)].reset_index(drop=True)
    
    train_x, train_y = np.array(train_df['mfccs_feature'].tolist()), np.array(train_df['class'].tolist())
    test_x, test_y = np.array(test_df['mfccs_feature'].tolist()), np.array(test_df['class'].tolist())
    
    # Build model
    with strategy.scope():    
        model = get_model(input_shape=train_x.shape[1:], learning_rate=LR_START, dropout=DROPOUT, isAugmentationActivated=True)
        
    # Train model
    history = model.fit(
    train_x, train_y,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=0,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_recall', factor=np.sqrt(0.1), patience=25, verbose=0, mode='max', min_delta=0.0001, cooldown=0, min_lr=0.5e-7),
        tf.keras.callbacks.ModelCheckpoint(F'{path}/src/models/{model_name}_fold{i+1}.h5', monitor='val_recall', verbose=0, save_best_only=True, mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_recall', patience=50, verbose=0, mode='max'),
    ])

    # Save history
    history = pd.DataFrame(history.history)
    history.to_csv(F'{path}/src/logs/{model_name}_fold{i+1}.csv', index=False)
    
    # Predict   
    with strategy.scope():    
        model = get_model(input_shape=train_x.shape[1:], learning_rate=LR_START, dropout=DROPOUT, isAugmentationActivated=False)
        model.load_weights(F'{path}/src/models/{model_name}_fold{i+1}.h5') 
    pred_y = model.predict(test_x)
    pred_y = np.argmax(pred_y, axis=1)
    result = pd.DataFrame({'pred': pred_y + 1, 'id': test_df['ID'].tolist(), 'true': test_df['Disease category'].tolist()})
    results = pd.concat([results, result], axis=0)
    
    result_recall = recall_score(result['true'], result['pred'], average=None)
    print(F"Test UAR (Fold {i+1}) :", round(result_recall.mean(), 4))
    
    # Load history
    log = pd.read_csv(F'{path}/src/logs/{model_name}_fold{i+1}.csv')
    print('epochs: ',len(log))
    plt.figure(figsize=(12, 4))

    # Summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(log['recall'])
    plt.plot(log['val_recall'])
    plt.title('recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'vaild'], loc='upper left') 

    # Summarize history for loss 
    plt.subplot(1, 2, 2)
    plt.plot(log['loss']) 
    plt.plot(log['val_loss']) 
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'vaild'], loc='upper left') 

    # Save image
    plt.savefig(F'{path}/src/logs/{model_name}_fold{i+1}.jpg')
    
results_recall = recall_score(results['true'], results['pred'], average=None)
print("Test UAR (All) :", round(results_recall.mean(), 4))
ConfusionMatrixDisplay(confusion_matrix(results['true'], results['pred'])).plot(cmap='Blues')

# %% [markdown]
# ## Predicting & Submission
# 

# %%
# public_df = pd.read_csv(F'{path}/src/data/test_datalist_public.csv')
# public_df['wav_path'] = public_df['ID'].apply(lambda x: F'{path}/src/data/public_voice_data/{x}.wav')
# print("public_df.columns :", public_df.columns)

# %%
public_df['mfccs_feature'] = public_df['wav_path'].apply(lambda x: audio_to_mfccs(x, n_mfcc=n_mfcc))
public_df['length'] = public_df['mfccs_feature'].apply(lambda x: x.shape[0])

# %%
group_by_length = public_df.groupby('length').size().reset_index(name='counts')
max_length = group_by_length['length'].max()
group_by_length

# %%
public_df['mfccs_feature'] = public_df['mfccs_feature'].apply(lambda x: np.concatenate((x, padded_mfcc[:max_length - x.shape[0]])))

# %%
pred_yy = []
for i in range(0, n_splits):    
    public_x = np.array(public_df['mfccs_feature'].tolist())
    with strategy.scope():    
        model = get_model(input_shape=public_x.shape[1:], learning_rate=LR_START, dropout=DROPOUT, isAugmentationActivated=False)
        model.load_weights(F'{path}/src/models/{model_name}_fold{i+1}.h5') 
    pred_y = model.predict(public_x)
    pred_yy.append(pred_y)

# %%
public_df['pred'] = list(np.argmax(np.array(pred_yy).mean(axis=0), axis=1) + 1)
public_df.head(5)

# %%
public_df[['ID', 'pred']].to_csv(F"{path}/src/submissions/{model_name}.csv", index=False, header=False)

# %%



