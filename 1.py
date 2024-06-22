import pandas as pd
import numpy as np
# 数据加载
train = pd.read_csv(r'D:\das\train .csv')
test = pd.read_csv(r'D:\das\test.csv')
submission = pd.read_csv(r'D:\das\submission.csv')
print(train.shape,test.shape,submission.shape)
# 训练集/测试集合并
df = pd.concat([train,test],axis=0)
# 选取类别特征
cat_columns = df.select_dtypes(include='object')
# 数据预处理-标签编码
from sklearn.preprocessing import LabelEncoder
job_le = LabelEncoder()
df['job'] = job_le.fit_transform(df['job'])
df['job'].value_counts()
df['marital'].value_counts()
df['marital'] = df['marital'].map({'unknow':0,'single':1,'married':2,'divorced':3})
df['marital'].value_counts()
df['education'].value_counts()
df['education'] = df['education'].map({'universiyu.degree':0,\
                                       'high.school':1,\
                                       'basic.9y':2,\
                                       'professional.course':3,\
                                       'basic.4y':4,\
                                       'basic.6y':5,\
                                       'unknown':6,\
                                       'illiterate':7})
df['education'].value_counts()
df['housing'].value_counts()
df['housing'] = df['housing'].map({'unknown':0,'no':1,'yes':2})
df['housing'].value_counts()
df['loan'] = df['loan'].map({'unknown':0,'no':1,'yes':2})
df['loan'].value_counts()
df['contact'] = df['contact'].map({'cellular': 0, 'telephone': 1})
df['contact'].value_counts()
# mon: 0, tue: 1, wed: 2, thu: 3, fri: 4
df['day_of_week'].value_counts()
df['day_of_week'] = df['day_of_week'].map({'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4})
df['day_of_week'].value_counts()
df['poutcome'] = df['poutcome'].map({'nonexistent': 0, 'failure': 1, 'success': 2})
df['poutcome'].value_counts()
df['default'].value_counts()
df['default'] = df['default'].map({'unknown': 0, 'no': 1, 'yes': 2})
df['default'].value_counts()
df['month'] = df['month'].map({'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, \
                 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
df['month'].value_counts()
df['subscribe'] = df['subscribe'].map({'no': 0, 'yes': 1})
df['subscribe'].value_counts()
# 切分数据集
train = df[df['subscribe'].notnull()]
test = df[df['subscribe'].isnull()]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
model_lgb = lgb.LGBMClassifier(num_leaves=2**5-1, reg_alpha=0.5, reg_lambda=0.25,
                               objective='binary',max_depth=-1, learning_rate=0.005,
                               min_child_samples=3, random_state=2022, n_estimators=2000,
                               subsample=1, colsample_bytree=1)
model_lgb.fit(train.drop('subscribe', axis=1), train['subscribe'])
from lightgbm import plot_importance
plot_importance(model_lgb)
y_pred = model_lgb.predict(test.drop('subscribe', axis=1))
result = pd.read_csv(r'D:\das\submission.csv')
result['subscribe'].value_counts()
subscribe_map = {1:'yes',0:'no'}
result['subscribe'] = [subscribe_map[x] for x in y_pred]
result.to_csv(r'D:\das\baseline_lgb1.csv',index=False)

