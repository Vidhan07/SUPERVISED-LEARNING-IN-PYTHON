import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X_train=pd.read_csv("C:/Users/hp-pc/Desktop/Data Analytics Project/Dataset_3.csv",low_memory=False)
X_label=pd.read_csv("C:/Users/hp-pc/Desktop/Data Analytics Project/Datatest_3.csv",low_memory=False)

model = Sequential()
model.add(Dense(200, input_dim=45, activation='sigmoid'))
model.add(Dense(70, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, X_label, epochs=500, batch_size=60, verbose=1)

_,accuracy=model.evaluate(X_train,X_label,verbose=1)

print('Accuracy %.2f'%(accuracy*100))