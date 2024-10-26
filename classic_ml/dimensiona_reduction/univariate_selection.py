from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

X, y = load_iris(return_X_y=True)

X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)

model = LogisticRegression()
cv = StratifiedKFold(n_splits=3)


scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
print('All features accuracy %.3f, std %.3f' % (np.mean(scores), np.std(scores)))

scores = cross_val_score(model, X_new, y, cv=cv, n_jobs=-1)
print('Selected features accuracy %.3f, std %.3f' % (np.mean(scores), np.std(scores)))