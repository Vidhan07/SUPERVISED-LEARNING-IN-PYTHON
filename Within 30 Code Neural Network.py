import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X_train_x=pd.read_csv("Patients_admit_30_days.csv",low_memory=False)
X_label=pd.read_csv("Patients_admit_30_days_labels.csv",low_memory=False)
X_test_x=pd.read_csv("X_test_less_than_30.csv")
X_test_patient=X_test_x['patient_nbr'].copy()
patients = X_train_x['patient_nbr2'].copy()
X_train = X_train_x.drop(['patient_nbr2'], 1)
X_test=X_test_x.drop(['patient_nbr'],1)
model = Sequential()
x = 250
y = 140
ep = 380
b_s = 180
model.add(Dense(x, input_dim=42, activation='relu'))
model.add(Dense(y, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, X_label, epochs=ep, batch_size=b_s, verbose=1)

_,accuracy=model.evaluate(X_train,X_label,verbose=0)

print('Accuracy %.2f'%(accuracy*100),'\nHidden Layer 1 Neurons',x,'\nHidden Layer 2 Neurons',y,'\nEpochs',ep,
      '\nBatch Size',b_s)

new = model.predict_classes(X_test)
patients = patients.to_numpy()
#print(new.shape)
X_test_patient = np.asarray(X_test_patient).reshape(-1, 1)
output_2 = np.append(X_test_patient, new, 1) + 1
np.savetxt('Output_2_Within_30_days.csv', output_2, delimiter=',')

