from rdflib import Graph, Namespace
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from para_tuning import tune_logistic_regression, tune_svm

# Define the RDF namespaces
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
dbpedia = Namespace("http://dbpedia.org/ontology/")

# Create an RDF graph and load the data
g = Graph()
g.parse("Nobel2012-2022-dataChanged.ttl", format="turtle")

# Prepare lists for labels and features
labels = []
features = []

# Iterate over the RDF graph and extract labels and features
for subject in g.subjects():
    if (subject, rdf.type, dbpedia.City) in g:
        label = g.value(subject, rdfs.label)
        labels.append(str(label))
        features.append(str(subject))

# Convert labels and features to NumPy arrays
Y = np.array(labels)
X = np.array(features)

# Feature extraction
vectorizer = TfidfVectorizer()
X_feature = vectorizer.fit_transform(X)

# Model selection and training
X_train, X_test, y_train, y_test = train_test_split(X_feature, Y, test_size=0.20, random_state=0)

# Logistic Regression with Hyperparameter Tuning
best_lr_model, best_lr_params = tune_logistic_regression(X_train, y_train)
y_pred_lr = best_lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr * 100)

# Prediction using Logistic Regression
new_data = ['new instance to classify']
new_data_feature = vectorizer.transform(new_data)
prediction_lr = best_lr_model.predict(new_data_feature)
print("Logistic Regression Prediction:", prediction_lr)

# SVM with Hyperparameter Tuning
best_svm_model, best_svm_params = tune_svm(X_train, y_train)
y_pred_svm = best_svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm * 100)

# Prediction using SVM
prediction_svm = best_svm_model.predict(new_data_feature)
print("SVM Prediction:", prediction_svm)
