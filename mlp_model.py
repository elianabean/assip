# %%
import os
import sys
import glob
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from collections import Counter
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
# %%
dir = "C:/Users/ewang/Downloads/Data_Set_for_Understanding_User_Identification_in_Virtual_Reality_Through_Behavioral_Biometrics_and_the_Effect_of_Body_Normalization"
file_pattern = os.path.join(dir, '*.csv')

data = []

for file_path in glob.glob(file_pattern):
  file_name = os.path.basename(file_path)
  if file_name.startswith('Archery') and 'repetition1.' in file_name and ('p1_' in file_name or 'p2_' in file_name) and 'BothNormalizations' in file_name:
    df = pd.read_csv(file_path)

    data.append(df)

df = pd.concat(data, ignore_index=True)

pd.set_option("display.max_rows", 15)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', 1000)

print('Shape of the dataset: ' + str(df.shape))
with pd.option_context('display.max_seq_items', None):
    print(df.head())
    print(df.tail())

# %%
def process_data(data, scenario):
  window_size = 10
  step_size = 1
  #split by bowling vs archery
  scenario_data = data[data['Scenario'] == scenario]

  #split by session, training = session 1 and test = session 2
  train_data = scenario_data[scenario_data['study_session'] == 1]
  test_data = scenario_data[scenario_data['study_session'] == 2]
  #test_data = train_data

  def create_windows_and_labels(data):
      X, y = [], []

      for person, person_data in data.groupby(['ParticipantID']):
        for repetition, rep_data in data.groupby(['repetition']):
            #tempdata = data.drop(columns=['ParticipantID', 'Scenario', 'study_session', 'repetition', 'HeightNormalization', 'ArmLengthNormalization', 'timestamp_ms', 'phase'])
            tempdata = data[['CenterEyeAnchor_pos_X', 'CenterEyeAnchor_pos_Y', 'CenterEyeAnchor_pos_Z']].values #Filtering to only these values for now

            #print("before windowing")
            #print(data[['CenterEyeAnchor_pos_X', 'CenterEyeAnchor_pos_Y', 'CenterEyeAnchor_pos_Z']].head(15))

            rolling_sum = data.drop(columns=['ParticipantID', 'Scenario', 'study_session', 'repetition', 'HeightNormalization', 'ArmLengthNormalization', 'timestamp_ms', 'phase']).rolling(10).sum()
            rolling_sum = rolling_sum.dropna()
            #rolling_sum = rolling_sum.reset_index(drop=True)
            rolling_sum['ParticipantID'] = data['ParticipantID']
            #print("values", data2.values)

            for window in rolling_sum.values:
               X.append(window[:-1])
               y.append(window[-1])
      return X, y

  X_train, y_train = create_windows_and_labels(train_data)
  X_test, y_test = create_windows_and_labels(test_data)

  X_train = np.array(X_train)
  #X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3])
  #X_train.reshape(X_train.shape[0], -1)
  y_train = np.array(y_train)
  #y_train = to_categorical(y_train, num_classes=1) #needed for LSTM
  X_test = np.array(X_test)
  #X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3])
  #X_test.reshape(X_test.shape[0], -1)
  y_test = np.array(y_test)
  #y_test = to_categorical(y_test, num_classes=1) #needed for LSTM

  return X_train, y_train, X_test, y_test

for normalization in ['HeightNormalization']: #should update these to include WithoutNormalization and BothNormalizations
  for scenario in ['Archery']:
    print(f"{scenario}, {normalization}")
    X_train, y_train, X_test, y_test = process_data(df, scenario)

    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    

# %%
np.set_printoptions(threshold=sys.maxsize)
print(y_train)
print(X_train.shape)

# %%
model = Sequential([ 
    
    Flatten(input_shape=(X_train.shape[1],)), 
    
    Dense(256, activation='relu'),   
    Dense(64, activation='relu'),
    Dense(16, activation='softmax'),   
]) 


model.compile(optimizer=SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# %%

model.fit(X_train, y_train, epochs=100, batch_size=32)
# %%
