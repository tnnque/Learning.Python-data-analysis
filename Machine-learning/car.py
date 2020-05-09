import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

input_file = 'car.data.txt'
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

# string to number
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}
classifier = RandomForestClassifier(**params)
classifier.fit(X,y)

accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print("Accuracy of the classifier:" + str(round(100*accuracy.mean(), 2)) + "%")

input_data = (['vhigh', 'vhigh', '2', '2', 'small', 'low'])
input_data_encoded = [-1] * len(input_data)
label_encoder = label_encoder[:-1]
lshape = np.shape(label_encoder)
ishape = np.shape(input_data)
print(lshape)
print(ishape)
for i, item in enumerate(input_data):
    le = label_encoder[i]
    shape_ = np.shape(le)
    shape1_ = np.shape(input_data[i])
    # le = le.transform(input_data[i])
#     # print(le)
#     input_data_encoded[:i] = int(le.fit_transform(input_data[i]))
pass

# input_data_encoded = np.array(input_data_encoded
#
# output_class = classifier.predict(input_data_encoded)
# print("Output class:", label_encoder[-1].inverse_transform(output_class)[0])