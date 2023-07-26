from rdflib import Graph, Namespace
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

# Train classifiers for each parent node (Logistic Regression and SVM)
lr_classifiers = {}
svm_classifiers = {}

# Transform X_train and X_test using the same vectorizer
X_train_feature = vectorizer.fit_transform(X_train)
X_test_feature = vectorizer.transform(X_test)


# Iterate over unique parent nodes in the training set
unique_parent_nodes = set([y.rsplit('.', 1)[0] for y in y_train])
for parent_node in unique_parent_nodes:
    # Filter examples belonging to the current parent node
    parent_examples = [x for x, y in zip(X_train, y_train) if y.startswith(parent_node)]
    parent_labels = [y for y in y_train if y.startswith(parent_node)]
    
    # Skip training if there is only one unique class label or no examples
    if len(set(parent_labels)) <= 1 or len(parent_examples) == 0:
        # print(f"Skipping training for parent node '{parent_node}' as there is only one unique class label or no examples.")
        continue
    
    # Feature extraction for the current parent node
    parent_examples_feature = vectorizer.transform(parent_examples)
    
    # Create a new Logistic Regression classifier for each parent node
    lr_classifiers[parent_node] = LogisticRegression()
    lr_classifiers[parent_node].fit(parent_examples_feature, parent_labels)
    
    # Create a new SVM classifier for each parent node
    svm_classifiers[parent_node] = SVC()
    svm_classifiers[parent_node].fit(parent_examples_feature, parent_labels)

# Predict labels for the testing set and print accuracy and prediction for each classifier
print("Logistic Regression Classifier:")
for parent_node, lr_classifier in lr_classifiers.items():
    parent_examples_test = [x for x, y in zip(X_test, y_test) if y.startswith(parent_node)]
    if len(parent_examples_test) == 0:
        print(f"Skipping prediction for parent node '{parent_node}' as there are no testing examples.")
        continue
    
    parent_examples_test_feature = vectorizer.transform(parent_examples_test)
    parent_labels_test = [y for y in y_test if y.startswith(parent_node)]
    parent_pred_lr = lr_classifier.predict(parent_examples_test_feature)
    
    accuracy_lr = accuracy_score(parent_labels_test, parent_pred_lr)
    print(f"Parent Node: {parent_node}, Accuracy (Logistic Regression): {accuracy_lr*100:.2f}%")
    print("Predictions (Logistic Regression):", parent_pred_lr)
    
print("\nSVM Classifier:")
for parent_node, svm_classifier in svm_classifiers.items():
    parent_examples_test = [x for x, y in zip(X_test, y_test) if y.startswith(parent_node)]
    if len(parent_examples_test) == 0:
        print(f"Skipping prediction for parent node '{parent_node}' as there are no testing examples.")
        continue
    
    parent_examples_test_feature = vectorizer.transform(parent_examples_test)
    parent_labels_test = [y for y in y_test if y.startswith(parent_node)]
    parent_pred_svm = svm_classifier.predict(parent_examples_test_feature)
    
    accuracy_svm = accuracy_score(parent_labels_test, parent_pred_svm)
    print(f"Parent Node: {parent_node}, Accuracy (SVM): {accuracy_svm*100:.2f}%")
    print("Predictions (SVM):", parent_pred_svm)
