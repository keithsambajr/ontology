import numpy as np
from rdflib import Graph, Namespace
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

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
# Fit the vectorizer on X_train and transform X_train and X_test
X_train_feature = vectorizer.fit_transform(X_train)
X_test_feature = vectorizer.transform(X_test)

# Train classifier for each parent node
classifier = {}

# Iterate over unique parent nodes in the training set
unique_parent_nodes = set([y.rsplit('.', 1)[0] for y in y_train])
for parent_node in unique_parent_nodes:
    # Filter examples belonging to the current parent node
    parent_examples = [x for x, y in zip(X_train, y_train) if y.startswith(parent_node)]
    parent_examples_feature = vectorizer.transform(parent_examples)
    parent_labels = [y for y in y_train if y.startswith(parent_node)]
    
    # Skip training if there is only one unique class label
    if len(set(parent_labels)) <= 1:
        continue
    
    # Create a new classifier for each parent node
    classifier[parent_node] = LogisticRegression()
    
    # Train the classifier
    classifier[parent_node].fit(parent_examples_feature, parent_labels)
    
# Predict labels for the testing set
y_pred = []
y_test_filtered = []

# Iterate over unique parent nodes in the testing set
unique_parent_nodes = set([y.rsplit('.', 1)[0] for y in y_test])
for parent_node in unique_parent_nodes:
    # Filter examples belonging to the current parent node
    parent_examples = [x for x, y in zip(X_test, y_test) if y.startswith(parent_node)]
    parent_examples_feature = vectorizer.transform(parent_examples)

    # Skip prediction if the parent node is not present in the trained classifiers
    if parent_node not in classifier:
        continue
    
    # Predict labels for the parent node
    parent_pred = classifier[parent_node].predict(parent_examples_feature)
    y_pred.extend(parent_pred)
    y_test_filtered.extend([y for y in y_test if y.startswith(parent_node)])

# Calculate accuracy
accuracy = accuracy_score(y_test_filtered, y_pred)
print("Accuracy:", accuracy * 100)
