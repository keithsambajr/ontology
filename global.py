from rdflib import Graph, Namespace
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

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

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train_feature = vectorizer.fit_transform(X_train)
X_test_feature = vectorizer.transform(X_test)

# Train the logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_feature, y_train)

# Predict labels for the testing set
y_pred = classifier.predict(X_test_feature)

# Print the predictions
print("Predictions:")
for i, prediction in enumerate(y_pred):
    print(f"Example {i+1}: {prediction}")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

#new classifier

# Train the SVM classifier
classifier_svm = SVC()
classifier_svm.fit(X_train_feature, y_train)

# Predict labels for the testing set using SVM
y_pred_svm = classifier_svm.predict(X_test_feature)

# Print the predictions using SVM
print("Predictions using SVM:")
for i, prediction in enumerate(y_pred_svm):
    print(f"Example {i+1}: {prediction}")

# Calculate accuracy using SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy using SVM:", accuracy_svm * 100)