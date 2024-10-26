from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)

rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
model = DecisionTreeClassifier(max_depth=10)
pipeline = Pipeline(steps=[('selector', rfe), ('clf', model)])
cv = StratifiedKFold(n_splits=5)

scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
print('All features accuracy %.3f, std %.3f' % (np.mean(scores), np.std(scores)))

scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)
print('Selected features accuracy %.3f, std %.3f' % (np.mean(scores), np.std(scores)))




