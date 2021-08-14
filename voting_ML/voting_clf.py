from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

rf_clf = RandomForestClassifier()
log_clf = LogisticRegression()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('lr', log_clf), ('svm', svm_clf)],
    voting='hard'
)

for clf in (rf_clf, log_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
