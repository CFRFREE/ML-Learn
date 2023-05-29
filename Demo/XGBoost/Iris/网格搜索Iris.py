import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

iris = load_iris()
X, y = iris.data, iris.target
col = iris.target_names
train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)  # 分训练集和验证集
parameters = {
	'max_depth': [1, 3, 5, 7, 9],
	'learning_rate': [0.01, 0.02, 0.05, 0.1],
	'n_estimators': [5, 10, 20, 30],
	'min_child_weight': [0, 2, 5, 10],
	'max_delta_step': [0, 0.2, 0.6, 1],
	'subsample': [0.6, 0.7, 0.8, 0.85],
	'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
	'reg_alpha': [0, 0.25, 0.5, 0.75],
	'reg_lambda': [0.2, 0.4, 0.6, 0.8],
}

xlf = xgb.XGBClassifier(max_depth=4,
                        learning_rate=0.01,
                        n_estimators=10,
                        objective='multi:softmax',
                        num_class=3,
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        seed=0,
                        )

gs = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3, n_jobs=-1)
gs.fit(train_x, train_y)

print("Best score: %0.3f" % gs.best_score_)
print("Best parameters set: %s" % gs.best_params_)
