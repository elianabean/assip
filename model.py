# %%
import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

# %%
dir = "C:/Users/ewang/Downloads/Data_Set_for_Understanding_User_Identification_in_Virtual_Reality_Through_Behavioral_Biometrics_and_the_Effect_of_Body_Normalization"
file_pattern = os.path.join(dir, '*.csv')

data = []

for file_path in glob.glob(file_pattern):
  file_name = os.path.basename(file_path)
  if file_name.startswith('Archery') and 'session1' in file_name and 'repetition1' in file_name and ('p1_' in file_name or 'p2_' in file_name) and 'BothNormalizations' in file_name:
    df = pd.read_csv(file_path)

    data.append(df)

df = pd.concat(data, ignore_index=True)

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
  #test_data = scenario_data[scenario_data['study_session'] == 2]
  test_data = train_data

  def create_windows_and_labels(data):
      X, y = [], []
      for person, person_data in data.groupby(['ParticipantID']):
        for repetition, rep_data in data.groupby(['repetition']):
            #tempdata = data.drop(columns=['ParticipantID', 'Scenario', 'study_session', 'repetition', 'HeightNormalization', 'ArmLengthNormalization', 'timestamp_ms', 'phase'])
            tempdata = data[['CenterEyeAnchor_pos_X', 'CenterEyeAnchor_pos_Y', 'CenterEyeAnchor_pos_Z']]
            windows = sliding_window_view(tempdata, window_shape=(window_size, tempdata.shape[1]))[::step_size]
            X.extend(windows)
            y.extend([person[0]-1] * len(windows))
      return X, y

  X_train, y_train = create_windows_and_labels(train_data)
  X_test, y_test = create_windows_and_labels(test_data)

  X_train = np.array(X_train)
  X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3])
  #print("X_train", X_train[:5])
  y_train = np.array(y_train)
  #print(y_train[:5])
  y_train_encoded = to_categorical(y_train, num_classes=2)
  #print(y_train_encoded)
  #print("y_train", y_train[:5])
  X_test = np.array(X_test)
  X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3])
  y_test = np.array(y_test)
  y_test_encoded = to_categorical(y_test, num_classes=2)

  return X_train_reshaped, y_train_encoded, X_test_reshaped, y_test_encoded

for normalization in ['HeightNormalization']: #should update these to include WithoutNormalization and BothNormalizations
  for scenario in ['Archery']:
    print(f"{scenario}, {normalization}")
    X_train, y_train, X_test, y_test = process_data(df, scenario)


    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")

# %%
model = Sequential()

model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(LSTM(100, activation='sigmoid', return_sequences=True))

model.add(LSTM(100, activation='sigmoid'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_train, y_train))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# %%
#model.predict(X_train)
# %%
