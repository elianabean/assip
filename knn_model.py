# %%
import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold


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
  if file_name.startswith('Archery') and 'BothNormalizations' in file_name:
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
    num_samples = []
    
    for person, person_data in data.groupby(['ParticipantID']):
      for repetition, rep_data in person_data.groupby(['repetition']):
        num_samples.append(len(rep_data))
        #create n-10+1 time series each of them has length=10 and label = person
        #temp_data = rep_data[['CenterEyeAnchor_pos_X', 'CenterEyeAnchor_pos_Y', 'CenterEyeAnchor_pos_Z']].values
        temp_data = rep_data.drop(columns=['ParticipantID', 'Scenario', 'study_session', 'repetition', 'HeightNormalization', 'ArmLengthNormalization', 'timestamp_ms', 'phase']).values
        for i in range(0, len(temp_data) - window_size + 1, step_size):
          window = temp_data[i:i + window_size]
          X.append(window)
          y.append(person[0]-1)

    return X, y, num_samples

  X_train, y_train, num_samples_train = create_windows_and_labels(train_data)
  X_test, y_test, num_samples_test = create_windows_and_labels(test_data)

  X_train = np.array(X_train)
  y_train = np.array(y_train)
  X_test = np.array(X_test)
  y_test = np.array(y_test)

  return X_train, y_train, X_test, y_test, num_samples_train, num_samples_test

for normalization in ['BothNormalizations']: #should update these to include WithoutNormalization and BothNormalizations
  for scenario in ['Archery']:
    print(f"{scenario}, {normalization}")
    X_train, y_train, X_test, y_test, num_samples_train, num_samples_test = process_data(df, scenario)

    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    

# %%
np.set_printoptions(threshold=sys.maxsize)
print(y_train)
print(X_train)

# %%
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#print(X_train[0:2])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
for n in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f'n_neighbors={n}, Validation Accuracy: {accuracy}')

# %%
knn = KNeighborsClassifier(n_neighbors = 7) 

knn.fit(X_train, y_train) 

y_pred = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100}%')

# %%
np.set_printoptions(threshold=sys.maxsize)
new_num_samples_test = num_samples_test.copy()

predictions = knn.predict(X_test)
print("predictions", predictions)
overall_labels_pred = []
overall_labels_true = []

new_num_samples_test.insert(0,0)
print("new_num_samples_test", new_num_samples_test)

for i in range(1, len(new_num_samples_test)):
  lst_pred = predictions[sum(new_num_samples_test[:i])-9*(i-1):sum(new_num_samples_test[:i])+new_num_samples_test[i]-9*i]
  lst_true = y_test[sum(new_num_samples_test[:i])-9*(i-1):sum(new_num_samples_test[:i])+new_num_samples_test[i]-9*i]
  overall_labels_pred.append(np.argmax(np.bincount(lst_pred)))
  overall_labels_true.append(np.argmax(np.bincount(lst_true)))

print("overall_labels_pred", overall_labels_pred)
print("overall_labels_true", overall_labels_true)
print(np.mean(np.array(overall_labels_pred) == np.array(overall_labels_true)))
