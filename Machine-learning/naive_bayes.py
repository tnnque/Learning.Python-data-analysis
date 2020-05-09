import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from logistic_regression import plot_classifier

# input_file = 'data_multivar_2.txt'
# X = []
# y = []
# with open(input_file, 'r') as f:
#     for line in f.readlines():
#         data = [float(x) for x in line.split(',')]
#         X.append(data[:-1])
#         y.append(data[-1])
# X = np.array(X)
# y = np.array(y)
#
# classifier_gaussiannb = GaussianNB()
# classifier_gaussiannb.fit(X, y)
# y_pred = classifier_gaussiannb.predict(X)
#
# accuracy = 100.0 * sum(y == y_pred) / X.shape[0]
# print("Accuracy of the classifier =", round(accuracy, 2), "%")
# plot_classifier(classifier_gaussiannb, X, y)
#
# from sklearn.model_selection import train_test_split, cross_val_score
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
# classifier_gaussiannb_new = GaussianNB()
# classifier_gaussiannb_new.fit(X_train, y_train)
# y_test_pred = classifier_gaussiannb_new.predict(X_test)
# accuracy = 100 * sum(y_test == y_test_pred) / X_test.shape[0]
# print("Accuracy of the classifier =", round(accuracy, 2), "%")
# plot_classifier(classifier_gaussiannb_new, X_test, y_test)
#
# num_validations = 5
# accuracy = cross_val_score(classifier_gaussiannb, X, y, scoring= 'accuracy', cv= num_validations)
# print("Accuracy:" + str(round(100 * accuracy.mean(), 2)) + "%")
#
# f1 = cross_val_score(classifier_gaussiannb, X, y, scoring= 'f1_weighted', cv= num_validations)
# print("F1:" + str(round(100 * f1.mean(), 2)) + "%")
#
# precision = cross_val_score(classifier_gaussiannb, X, y, scoring= 'precision_weighted', cv= num_validations)
# print("Precision:" + str(round(100 * precision.mean(), 2)) + "%")
#
# recall = cross_val_score(classifier_gaussiannb, X, y, scoring= 'recall_weighted', cv= num_validations)
# print("Recall:" + str(round(100 * recall.mean(), 2)) + "%")

from sklearn.metrics import confusion_matrix
y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
confusion_mat = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat, interpolation= 'nearest', cmap= plt.cm.Paired)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(confusion_mat)

# from sklearn.metrics import classification_report
# target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
# print(classification_report(y_true, y_pred, target_names= target_names))
