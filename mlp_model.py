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
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD
# %%
dir = "C:/Users/ewang/Downloads/Data_Set_for_Understanding_User_Identification_in_Virtual_Reality_Through_Behavioral_Biometrics_and_the_Effect_of_Body_Normalization"
file_pattern = os.path.join(dir, '*.csv')

def process(df):
  #we deduct the global coordinates of the hand-held controllers from the HMD
  df['LeftControllerAnchor_pos_X'] = df['LeftControllerAnchor_pos_X'] - df['CenterEyeAnchor_pos_X']
  df['LeftControllerAnchor_pos_Y'] = df['LeftControllerAnchor_pos_Y'] - df['CenterEyeAnchor_pos_Y']
  df['LeftControllerAnchor_pos_Z'] = df['LeftControllerAnchor_pos_Z'] - df['CenterEyeAnchor_pos_Z']

  df['RightControllerAnchor_pos_X'] = df['RightControllerAnchor_pos_X'] - df['CenterEyeAnchor_pos_X']
  df['RightControllerAnchor_pos_Y'] = df['RightControllerAnchor_pos_Y'] - df['CenterEyeAnchor_pos_Y']
  df['RightControllerAnchor_pos_Z'] = df['RightControllerAnchor_pos_Z'] - df['CenterEyeAnchor_pos_Z']

  #we subtract the global coordinates of the initial point of appearance from all subsequently captured points.
  initial_HMD_X = df['CenterEyeAnchor_pos_X'][0]
  initial_HMD_Y = df['CenterEyeAnchor_pos_Y'][0]
  initial_HMD_Z = df['CenterEyeAnchor_pos_Z'][0]

  df['CenterEyeAnchor_pos_X'] = df['CenterEyeAnchor_pos_X'] - initial_HMD_X
  df['CenterEyeAnchor_pos_Y'] = df['CenterEyeAnchor_pos_Y'] - initial_HMD_Y
  df['CenterEyeAnchor_pos_Z'] = df['CenterEyeAnchor_pos_Z'] - initial_HMD_Z

  #normalize the Euler angle values in an interval of [0, 1)
  def normalize(angle):
      return (angle % 360) / 360.0
  
  df['CenterEyeAnchor_euler_X'] = df['CenterEyeAnchor_euler_X'].apply(normalize)
  df['CenterEyeAnchor_euler_Y'] = df['CenterEyeAnchor_euler_Y'].apply(normalize)
  df['CenterEyeAnchor_euler_Z'] = df['CenterEyeAnchor_euler_Z'].apply(normalize)
  df['LeftControllerAnchor_euler_X'] = df['LeftControllerAnchor_euler_X'].apply(normalize)
  df['LeftControllerAnchor_euler_Y'] = df['LeftControllerAnchor_euler_Y'].apply(normalize)
  df['LeftControllerAnchor_euler_Z'] = df['LeftControllerAnchor_euler_Z'].apply(normalize)
  df['RightControllerAnchor_euler_X'] = df['RightControllerAnchor_euler_X'].apply(normalize)
  df['RightControllerAnchor_euler_Y'] = df['RightControllerAnchor_euler_Y'].apply(normalize)
  df['RightControllerAnchor_euler_Z'] = df['RightControllerAnchor_euler_Z'].apply(normalize)

  return df

data = []

for file_path in glob.glob(file_pattern):
  file_name = os.path.basename(file_path)
  if file_name.startswith('Archery') and 'repetition1.' in file_name and 'BothNormalizations' in file_name:
    df = pd.read_csv(file_path)
    df = process(df)

    data.append(df)

df = pd.concat(data, ignore_index=True)

pd.set_option("display.max_rows", 15)
pd.set_option('display.max_columns', 40)
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
  #filter by bowling vs archery
  scenario_data = data[data['Scenario'] == scenario]

  #split by session, training = session 1 and test = session 2
  train_data = scenario_data[scenario_data['study_session'] == 1]
  test_data = scenario_data[scenario_data['study_session'] == 2]

  def create_windows_and_labels(data):
    X, y = [], []
    
    for person, person_data in data.groupby(['ParticipantID']):
      #create n-10+1 time series each of them has length=10 and label = person
      #temp_data = person_data[['CenterEyeAnchor_pos_X', 'CenterEyeAnchor_pos_Y', 'CenterEyeAnchor_pos_Z']].values
      temp_data = person_data.drop(columns=['ParticipantID', 'Scenario', 'study_session', 'repetition', 'HeightNormalization', 'ArmLengthNormalization', 'timestamp_ms', 'phase']).values
      for i in range(0, len(temp_data) - window_size + 1, step_size):
        window = temp_data[i:i + window_size]
        X.append(window)
        y.append(person[0]-1)

    return X, y

  X_train, y_train = create_windows_and_labels(train_data)
  X_test, y_test = create_windows_and_labels(test_data)

  X_train = np.array(X_train)
  y_train = np.array(y_train)
  X_test = np.array(X_test)
  y_test = np.array(y_test)

  return X_train, y_train, X_test, y_test

for normalization in ['BothNormalizations']: #should update these to include WithoutNormalization and BothNormalizations
  for scenario in ['Archery']:
    print(f"{scenario}, {normalization}")
    X_train, y_train, X_test, y_test = process_data(df, scenario)

    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    

# %%
np.set_printoptions(threshold=sys.maxsize)
print(y_train[-1])
print(X_train[-1])

# %%
model = Sequential([ 
    Flatten(input_shape=(X_train.shape[1],X_train.shape[2])), 
    
    Dense(256, activation='relu'),
    Dense(64, activation='relu'),
    Dense(16, activation='softmax'),
]) 

print(model.summary())

model.compile(optimizer=SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# %%
model.fit(X_train, y_train, epochs=100, batch_size=32)

results = model.evaluate(X_test, y_test, batch_size=32)
print("test loss, test acc:", results)
