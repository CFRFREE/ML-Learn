from sklearn import svm
from sklearn import datasets
import joblib

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
joblib.dump(clf, 'test.pkl')
clf_new = joblib.load('test.pkl')
print(clf_new.predict(X[0:1]))
