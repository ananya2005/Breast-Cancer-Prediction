from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split

cancer = datasets.load_breast_cancer()

print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)
cancer.data.shape

print(cancer.data[0:5])
print(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

report_svc = classification_report(y_test, y_pred)
print(report_svc)

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print(metrics.confusion_matrix(y_test, y_pred))
