# 调包
import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train["Survived"]
train = train.drop(["Survived"], axis=1)
all_data = pd.concat([train, test])
PassengerId = test["PassengerId"]
all_data = all_data.drop(["PassengerId"], axis=1)
print('训练集和测试集共有{0}个样本和{1}个特征。'.format(all_data.shape[0], all_data.shape[1]))
print("****************")
print(all_data.head())
print("****************")

print(all_data.isnull().sum().sort_values(ascending=False))
print("****************")
all_data["Cabin"] = all_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# 获取空值的index
index_NaN_age = list(all_data["Age"][all_data["Age"].isnull()].index)

# 默认用随机数填充
age_mean = all_data["Age"].mean()
age_std = all_data["Age"].std()
age_random_list = np.random.randint(age_mean - age_std, age_mean + age_std, size=len(index_NaN_age))
all_data["Age"][all_data["Age"].isnull()] = age_random_list

# 当存在类似情况时，用类似样本的中位数填充空值
for i in index_NaN_age:
	same_samples_age = all_data["Age"][(
			(all_data["SibSp"] == all_data["SibSp"].iloc[i]) & (all_data["Parch"] == all_data["Parch"].iloc[i]) & (
			all_data["Pclass"] == all_data["Pclass"].iloc[i]))].median()
	if not np.isnan(same_samples_age):
		all_data["Age"].iloc[i] = same_samples_age

# 检查是否完全填充
assert all_data["Age"].isnull().sum() == 0

# 筛出两条缺失值
print(all_data[all_data["Embarked"].isnull()])
print("****************")
all_data[all_data["Pclass"] == 1].groupby("Embarked")["Fare"].describe()
all_data["Embarked"] = all_data["Embarked"].fillna("S")
fare_med = all_data["Fare"][(all_data["Embarked"] == 'S') & (all_data["Pclass"] == 3)].median()
all_data["Fare"] = all_data["Fare"].fillna(fare_med)

# 确认所有缺失值处理完毕
print(all_data.isnull().sum())
print("****************")


def getTitle(name):
	title_search = re.search('([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""


all_data["Title"] = all_data["Name"].apply(getTitle)

# 查看各种称谓出现的次数
all_data["Title"].value_counts()
rare_title = ["Dr", "Rev", "Col", "Major", "Don", "Jonkheer", "Sir", "Lady", "Countess", "Capt", "Dona"]

all_data["Title"].replace(rare_title, "Rare", inplace=True)
all_data["Title"].replace(["Mlle", "Ms"], "Miss", inplace=True)
all_data["Title"].replace(["Mme"], "Mrs", inplace=True)

# 检查是否完成合并
print(all_data["Title"].value_counts())
print("****************")

all_data["FamilySize"] = all_data["SibSp"] + all_data["Parch"] + 1
all_data["IsAlone"] = 0
all_data.loc[all_data["FamilySize"] == 1, "IsAlone"] = 1

# 查看是否为一人的人数
print(all_data["IsAlone"].value_counts())
print("****************")

all_data["AgeCut"] = pd.cut(all_data["Age"], 5)
all_data["AgeCut"].value_counts()

all_data["FareCut"] = pd.qcut(all_data["Fare"], 4)
all_data["FareCut"].value_counts()

# 年龄分段
all_data.loc[all_data["Age"] < 16, "AgeGroup"] = 0
all_data.loc[(all_data["Age"] >= 16) & (all_data["Age"] < 32), "AgeGroup"] = 1
all_data.loc[(all_data["Age"] >= 32) & (all_data["Age"] < 48), "AgeGroup"] = 2
all_data.loc[(all_data["Age"] >= 48) & (all_data["Age"] < 64), "AgeGroup"] = 3
all_data.loc[all_data["Age"] >= 64, "AgeGroup"] = 4

# 票价分段
all_data.loc[all_data["Fare"] < 7.896, "FareGroup"] = 0
all_data.loc[(all_data["Fare"] >= 7.896) & (all_data["Fare"] < 14.454), "FareGroup"] = 1
all_data.loc[(all_data["Fare"] >= 14.454) & (all_data["Fare"] < 31.275), "FareGroup"] = 2
all_data.loc[all_data["Fare"] >= 31.275, "FareGroup"] = 3

# 取整
all_data["AgeGroup"] = all_data["AgeGroup"].astype("int")
all_data["FareGroup"] = all_data["FareGroup"].astype("int")
drop_features = ["Name", "Age", "SibSp", "Parch", "Ticket", "Fare", "FamilySize", "AgeCut", "FareCut"]
all_data.drop(drop_features, axis=1, inplace=True)

# 查看结果
print(all_data.head())
print("****************")

# 性别转换为二值变量
all_data["Sex"] = all_data["Sex"].map({"female": 0, "male": 1}).astype(int)

# 整型转换为字符型
all_data["Pclass"] = all_data["Pclass"].astype("str")
all_data["AgeGroup"] = all_data["AgeGroup"].astype("str")
all_data["FareGroup"] = all_data["FareGroup"].astype("str")

# 虚拟变量
all_data = pd.get_dummies(all_data)

# 查看虚拟化后的变量名
print(all_data.columns)
print("****************")

# 分离训练集和测试集
X_train = all_data[:y.shape[0]]
X_test = all_data[y.shape[0]:]

models = {'LogisticRegression': LogisticRegressionCV(), 'NeuralNetwork': MLPClassifier(),
          'CART': DecisionTreeClassifier(), 'SVM': SVC(), 'KNN': KNeighborsClassifier(), 'NaiveBayes': BernoulliNB(),
          'RandomForest': RandomForestClassifier(), 'ExtraTree': ExtraTreesClassifier(),
          'AdaBoost': AdaBoostClassifier(), 'GBDT': GradientBoostingClassifier(), 'XGBoost': XGBClassifier(),
          'LightGBM': LGBMClassifier()}

kf = KFold(10)

for model in models:
	cv_result = cross_val_score(models[model], X_train, y, cv=kf, scoring='accuracy')
	print('%s模型的交叉验证得分平均值%.2f%%，标准差%.2f%%。' % (model, cv_result.mean() * 100, cv_result.std() * 100))
print("****************")

bagging_models = {'RandomForest': RandomForestClassifier(), 'ExtraTree': ExtraTreesClassifier()}
boosting_models = {'AdaBoost': AdaBoostClassifier(), 'GBDT': GradientBoostingClassifier(), 'XGBoost': XGBClassifier(),
                   'LightGBM': LGBMClassifier()}

bagging_params = {'n_estimators': [10, 50, 100, 200, 500, 800]}
boosting_params = {'n_estimators': [10, 50, 100, 200, 500, 800], 'learning_rate': [0.005, 0.01, 0.1, 1]}

kf = KFold(10)

for model in bagging_models:
	grid = GridSearchCV(estimator=bagging_models[model], param_grid=bagging_params, cv=kf, scoring='accuracy')
	grid_result = grid.fit(X_train, y)
	print('%s模型的最优参数是%s，得分%.2f%%。' % (model, grid_result.best_params_, grid_result.best_score_ * 100))

for model in boosting_models:
	grid = GridSearchCV(estimator=boosting_models[model], param_grid=boosting_params, cv=kf, scoring='accuracy')
	grid_result = grid.fit(X_train, y)
	print('%s模型的最优参数是%s，得分%.2f%%。' % (model, grid_result.best_params_, grid_result.best_score_ * 100))
