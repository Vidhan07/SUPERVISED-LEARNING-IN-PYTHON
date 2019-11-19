import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
from tensorflow_core.python.keras.distribute.keras_utils_test import Counter

def transform(X_train, des):

    X_train['service_utilization'] = X_train['number_outpatient'] + X_train['number_inpatient'] + X_train['number_emergency']

    feature_set = ['age', 'time_in_hospital', 'num_procedures', 'num_medications', 'number_diagnoses', 'metformin',
                   'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide',
                   'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'number_outpatient', 'number_emergency',
                   'number_inpatient', 'diag_2',
                   'tolazamide', 'insulin', 'glyburide-metformin', 'gender', 'A1Cresult', 'max_glu_serum', 'race',
                   'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1']

    # 'diag_1_0','diag_1_1','diag_1_2','diag_1_3','diag_1_4','diag_1_5','diag_1_6','diag_1_7','diag_1_8','admission_type_id_1','admission_type_id_2','admission_type_id_3','Asian','Others','AmericanAfrican','Hispanic','Caucasian'
    # ,'discharge_disposition_id_1','discharge_disposition_id_2','discharge_disposition_id_3','Admission_source_id_1','Admission_source_id_2','Admission_source_id_3','Admission_source_id_4','Admission_source_id_5','Admission_source_id_6'

    if (des == False):
        x_labels = X_train['readmitted']
    patients = X_train['patient_nbr'].to_numpy()
    X_train = X_train[feature_set]

    train_input_new = pd.DataFrame(X_train, columns=list(X_train.columns))

    if (des == True):
        return train_input_new, patients

    X_trai, X_test, y_trai, y_test = train_test_split(train_input_new, x_labels, test_size=0.30, random_state=0)

    rm = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=55, max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=10,
                                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                                oob_score=False, random_state=None, verbose=0,
                                warm_start=False)
    rm.fit(X_train, x_labels)

    rm_prd = rm.predict(X_trai)

    print("Train Data Accuracy")
    print("Accuracy is {0:.2f}".format(accuracy_score(y_trai, rm_prd)))
    print("Precision is {0:.2f}".format(precision_score(y_trai, rm_prd)))
    print("Recall is {0:.2f}".format(recall_score(y_trai, rm_prd)))

    return rm

X_train = pd.read_csv("C:/Users/hp-pc/Desktop/Data Analytics Project/Train_C2T1.csv", low_memory=False)
X_test = pd.read_csv("C:/Users/hp-pc/Desktop/Data Analytics Project/Data_Test.csv", low_memory=False)
rm = transform(X_train, False)
input_t, patien = transform(X_test, True)
pre = np.asarray(rm.predict(input_t)).reshape(-1, 1)
patien = patien.reshape(-1, 1).astype(int)
out = np.append(patien, pre, 1).astype(int)
np.savetxt('output.csv', out, delimiter=',')
print(out)
print(pre.shape)
print(patien.shape)

