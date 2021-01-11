import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.data')

x = np.array(df.iloc[:,0:4])
y = np.array(df.iloc[:,4:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(x_train,y_train)

from sklearn.metrics import accuracy_score
predicted = sv.predict(x_test)
print (accuracy_score(y_test, predicted))

pickle.dump(sv,open('iris.pkl','wb'))