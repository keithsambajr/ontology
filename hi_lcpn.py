import numpy as np
from sklearn.linear_model import LogisticRegression
from hiclass import LocalClassifierPerNode
from rdflib import Graph, Namespace
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load the data
g = Graph()
g.parse("Nobel2012-2022-dataChanged.ttl", format="turtle")

# Prepare lists for labels and features
X = []
y = []

# Iterate over the RDF graph and extract labels and features
for s, p, o in g.triples((None, None, None)):
    X.append(str(s))
    y.append(str(o))    

# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

print(y_train[:5])
print(y_train[:5])
# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()

# Fit and transform X_train
X_train_feature = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_feature = vectorizer.transform(X_test)

# Train classifiers using forest classifiers for logistic regression
rf = LogisticRegression()
classifier = LocalClassifierPerNode(rf, X_train_feature, y_train)
classifier.fit(X_train_feature, y_train)

# Predict labels for the testing set
predictions = classifier.predict(X_test_feature)
Accuracy = accuracy_score(y_test, predictions)*100