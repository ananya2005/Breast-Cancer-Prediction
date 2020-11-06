from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.metrics import classification_report

cancer = datasets.load_breast_cancer()

print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)

cancer.data.shape

print(cancer.data[0:5])
print(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
knn_model_1 = knn.fit(X_train, y_train)
y_true, y_pred = y_test, knn_model_1.predict(X_test)

print(classification_report(y_true, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

print(metrics.confusion_matrix(y_test, y_pred))
