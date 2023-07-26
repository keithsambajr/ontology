from rdflib import Graph, Namespace
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np

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
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Model evaluation
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy*100)

# Prediction
new_data = ['new instance to classify']
new_data_feature = vectorizer.transform(new_data)
prediction = classifier.predict(new_data_feature)

print("Prediction:", prediction)




#svm classifier
# Model selection and training
X_train, X_test, y_train, y_test = train_test_split(X_feature, Y, test_size=0.20, random_state=0)
classifier = SVC()
classifier.fit(X_train, y_train)

# Model evaluation
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy*100)

# Prediction
new_data = ['new instance to classify']
new_data_feature = vectorizer.transform(new_data)
prediction = classifier.predict(new_data_feature)

print("Prediction:", prediction)    










# C (inverse of regularization strength): The default value is 1.0. A smaller value of C corresponds to stronger regularization.

# penalty: The default value is 'l2', which indicates L2 regularization. Other options include 'l1' for L1 regularization and 'none' for no regularization.

# solver: The default solver depends on the problem size. For smaller problems, the default solver is 'liblinear', and for larger problems, the default solver is 'lbfgs'. Other solver options include 'newton-cg', 'sag', and 'saga'.

# max_iter: The default value is 100, which is the maximum number of iterations for the solver to converge.