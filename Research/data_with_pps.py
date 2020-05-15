import math, statistics
import csv
import numpy as np
import pandas as pd
import ppscore as pps
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Define dataset loading function
def data_load(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :11]
    y = df.iloc[:, -1:]

    return df, X, y

# Input and separate X, y for training and testing data
df, X, y = data_load('winequality-red.csv')
# data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
# X = pd.DataFrame(data_scaler.fit_transform(X), columns=df.columns[:11])
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Use Predictive Power Score to find patterns and select features
fig, ax = plt.subplots(figsize=(10, 6))
df_matrix = pps.matrix(df)
sns.heatmap(round(df_matrix,5), annot=True, vmin=0, vmax=1, ax=ax, cmap="BuGn", linewidths=0.5)
fig.subplots_adjust(top=0.5)
t= fig.suptitle('Predictive Power Score Heatmap', fontsize=14)
plt.show()

fig1, ax1 = plt.subplots(figsize=(10, 6))
correlation = df.corr()
sns.heatmap(round(correlation,5), annot=True, ax=ax1, cmap="coolwarm", linewidths=.5)
fig1.subplots_adjust(top=0.5)
t1= fig1.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)
plt.show()

classifiers = {}
parameters = {}

# Create and update dictionary with Extra Trees Ensemble
classifiers.update({"Extra Trees Ensemble": ExtraTreesClassifier()})
parameters.update({"Extra Trees Ensemble": {
                                            'classifier__n_estimators': [200],
                                            'classifier__class_weight': [None, 'balanced'],
                                            'classifier__max_features': ['auto', 'sqrt', 'log2'],
                                            'classifier__max_depth': [3, 4, 5, 6, 7, 8],
                                            'classifier__min_samples_split': [0.005, 0.01, 0.05, 0.10],
                                            'classifier__min_samples_leaf': [0.005, 0.01, 0.05, 0.10],
                                            'classifier__criterion':['gini', 'entropy']     ,
                                            'classifier__n_jobs': [-1]
                                             }})

# Create and update dictionary with Gradient Boosting
classifiers.update({"Gradient Boosting": GradientBoostingClassifier()})
parameters.update({"Gradient Boosting": {
                                        'classifier__learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],
                                        'classifier__n_estimators': [200],
                                        'classifier__max_depth': [2,3,4,5,6],
                                        'classifier__min_samples_split': [0.005, 0.01, 0.05, 0.10],
                                        'classifier__min_samples_leaf': [0.005, 0.01, 0.05, 0.10],
                                        'classifier__max_features': ['auto','sqrt', 'log2'],
                                        'classifier__subsample': [0.8, 0.9, 1]
                                         }})

# Create and update dictionary with Random Forest Classifier
classifiers.update({"Random Forest": RandomForestClassifier()})
parameters.update({"Random Forest": {
                                    'classifier__n_estimators': [200],
                                    'classifier__class_weight': [None, 'balanced'],
                                    'classifier__max_features': ['auto', 'sqrt', 'log2'],
                                    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
                                    'classifier__min_samples_split': [0.005, 0.01, 0.05, 0.10],
                                    'classifier__min_samples_leaf': [0.005, 0.01, 0.05, 0.10],
                                    'classifier__criterion':['gini', 'entropy'],
                                    'classifier__n_jobs': [-1]
                                     }})
                                    # this "Classifier Hyper-parameters" method is sourced from Frank Ceballos at https://gist.githubusercontent.com/frank-ceballos/0c7f2ecdb15854ff6b5e88af4bd01c41/raw/a5b8bed155b511c00dd8328d66fd1deadcbdf9b6/05ModelDesign&Selection.py

# ### Start the work with some 'feature importance' classifiers
# Create a dictionary for the classifiers
feature_importance = {'Extra Trees Ensemble', 'Gradient Boosting', 'Random Forest'}

def select_classifier(classifier_name):
    # Input the classifier we want to use from dictionary
    classifier = classifiers[classifier_name]

    # Scale features
    feature_scaler = preprocessing.StandardScaler()

    # Steps for pipeline and grid
    steps = [('scaler', feature_scaler), ('classifier', classifier)]
    pipeline = Pipeline(steps=steps)
    grid = parameters[classifier_name]

    # Create grid search object
    grid_search = GridSearchCV(pipeline, grid, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc')
    # Hieu fix
    from sklearn.preprocessing import LabelEncoder
    ohe = LabelEncoder()
    labels = np.array(y_train)
    ohe.fit(labels)
    labels = ohe.transform(np.array(labels))
    grid_search.fit(X_train, np.ravel(labels))

    # Get final parameters and score
    result_params = grid_search.best_params_
    result_score = grid_search.best_score_

    # Update classifier parameters
    tuned_params = {item[12:]: result_params[item] for item in result_params}
    classifier.set_params(**tuned_params)

    return classifier, tuned_params, feature_scaler

classifier, tuned_params, feature_scaler = select_classifier('Random Forest')

# Select Features using RFECV
class PipelineRFE(Pipeline):
    # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self
# Define pipeline for RFECV
steps = [('scaler', feature_scaler), ('classifier', classifier)]
pipe = PipelineRFE(steps = steps)

# Initialize RFECV object
feature_selector = RFECV(pipe, cv = 5, step = 1, scoring = 'roc_auc', verbose = 1)

# Fit RFECV
feature_selector.fit(X_train, np.ravel(y_train))

# Get selected features
feature_names = X_train.columns
selected_features = feature_names[feature_selector.support_].tolist()