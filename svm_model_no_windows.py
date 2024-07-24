# %%
import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from collections import Counter

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
    #split by bowling vs archery
    scenario_data = data[data['Scenario'] == scenario]

    #split by session, training = session 1 and test = session 2
    train_data = scenario_data[scenario_data['study_session'] == 1]
    test_data = scenario_data[scenario_data['study_session'] == 2]

    def create_windows_and_labels(data):
        X, y = [], []

        tempdata = data.drop(columns=['ParticipantID', 'Scenario', 'study_session', 'repetition', 'HeightNormalization', 'ArmLengthNormalization', 'timestamp_ms', 'phase'])
        X = tempdata.values
        y = data['ParticipantID'].values - 1
        
        return X, y

    X_train, y_train = create_windows_and_labels(train_data)
    X_test, y_test = create_windows_and_labels(test_data)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

for normalization in ['HeightNormalization']: #should update these to include WithoutNormalization and BothNormalizations
  for scenario in ['Archery']:
    print(f"{scenario}, {normalization}")
    X_train, y_train, X_test, y_test = process_data(df, scenario)
    
    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    

# %%
np.set_printoptions(threshold=sys.maxsize)
print(y_train[:100])
print(X_train[:100])

# %%
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
