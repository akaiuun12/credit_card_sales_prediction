# %% Project 2. NH Card Sales Prediction
# This is a replication project of credit card sales prediction. 
# How to make better
#  - At least 3 different models (RandomForest, XGBoost, LightGBM)
#  - Add more about Key Questions
#  - What I focused on this project

# %% 0. Environment Settings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

print(f'numpy version: {np.__version__}')
print(f'pandas version: {pd.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'matplotlib version: {mpl.__version__}')

plt.style.use('ggplot')

plt.rc('font', family='Malgun Gothic')
plt.rc('font', family='AppleGothic')
plt.rc('axes', unicode_minus=False)


# %% 1. LOAD THE DATA
# Data Source: KDX Data - [NH농협카드] 일자별 소비현황_서울

bas_ym = [
    '202001', '202002', '202003', '202004', '202005', '202006',
    '202007', '202008', '202009', '202010', '202011', '202012',
    '202101', '202102', '202103', '202104', '202105', '202106',
    # '202107', '202108', '202109', '202110', '202111', '202112', 
    '202301', '202302', '202303', '202304', '202305', '202306', 
    '202307', '202308', '202309', '202310', '202311', '202312', 
    '202401', '202402', '202403', '202404', '202405', '202406', 
]

df = pd.DataFrame()

for i, var in enumerate(bas_ym):
    data_path = f'data/[NH농협카드] 일자별 소비현황_서울_{var}.csv'
    
    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xbd in position 0: invalid start byte
    # UnicodeDecodeError: 'euc_kr' codec can't decode byte 0xec in position 0: illegal multibyte sequence
    try:
        tmp_df = pd.read_csv(data_path, encoding='utf-8-sig')
    except:
        tmp_df = pd.read_csv(data_path, encoding='euc-kr')
    
    df = pd.concat([df, tmp_df])
    
print(df.shape)
print(df.head())


# %% 2. DATA PREPROCESSING
# Type Conversion (int64 -> datetime64)
df['date'] = pd.to_datetime(df['승인일자'], format='%Y%m%d')

# Decimal Point Handling
df['이용금액_전체'] = df['이용금액_전체'] / 100
df['이용금액_개인'] = df['이용금액_개인'] / 100
df['이용금액_법인'] = df['이용금액_법인'] / 100

# Derived Variables
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

df['avg_sales'] = df['이용금액_전체'] * 1000000 / df['이용건수_전체'] / 1000
df['avg_sales_psn'] = df['이용금액_개인'] * 1000000 / df['이용건수_개인'] / 1000
df['avg_sales_cor'] = df['이용금액_법인'] * 1000000 / df['이용건수_법인'] / 1000

# Nominal to Ordinal Variable
df['dayname'] = pd.Categorical(df['date'].dt.day_name(), 
                               categories=['Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday'],
                               ordered=True)

# ※ ★★★ 공휴일 캘린더 추가 (Maybe Side Project?) ★★★

# Nullity Check
print(df.isna().sum())

# Train-Test Split
df_train = df[df['year'] != 2024]
df_test = df[df['year'] == 2024]

print(df_train.shape, df_test.shape)


# %% 3. EDA WITH VISUALIZATION
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

# 3-1. Null Handling
sns.heatmap(df.isna(), ax=axes[0])
axes[0].set_title('결측치 NULL 확인')
axes[0].set_yticklabels([])

# 3-2. Correlation Matrix (Multicolinearity Check)
corr_mat = df.drop(['시도', '승인일자', 'dayname'], axis=1).corr()

sns.heatmap(corr_mat, annot=True, fmt='.1f', ax=axes[1])
# sns.heatmap((abs(corr_mat) > 0.3), ax=axes[2])
axes[1].set_yticklabels([])
# axes[2].set_yticklabels([])

axes[1].set_title('변수 간 상관관계')
plt.tight_layout()
plt.show() 

# 3-3. Outlier Detection
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,9))

variables = ['이용금액_전체', '이용금액_개인', '이용금액_법인']
 
for i, vars in enumerate(variables):
    sns.lineplot(data=df, x='date', y=vars,
                ax=axes[i%3][0])
    sns.boxplot(data=df, x=vars,
                ax=axes[i%3][1])
    axes[i%3][0].set_title(vars)

    plt.tight_layout()

fig.suptitle('')
plt.show()

# # ※ Additional Analysis Needed
# sns.lineplot(data=df,
#              x='date', y='이용건수_법인')
# sns.scatterplot(data=df[df['이용금액_법인'] > 60000],
#                 x='date', y='이용건수_법인', c='b')
# plt.show()

# 3-4. Sales by Day
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

sns.barplot(data=df, x='dayname', y='이용건수_전체', 
            palette='crest', ax=axes[0])
sns.barplot(data=df, x='dayname', y='이용금액_전체', 
            palette='crest', ax=axes[1])
plt.tight_layout()
plt.show()


# %% 4. Regression Model Specification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from xgboost import 

# 4-1. Train-Validation Split
ind_vars = ['승인일자', 'year', 'month', 'day', 'dayofweek']
dep_vars = ['이용건수_개인', '이용건수_법인', '이용금액_개인', '이용금액_법인']

X = df[df['year'] != 2024][ind_vars]
y = df[df['year'] != 2024][dep_vars]

X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size=0.2, 
                                                      random_state=42
                                                      )
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

# 4-2. Machine Learning (RandomForest, XGBoost)
model_rf = RandomForestRegressor(n_estimators=1000,
                                 max_depth=20,
                                 random_state=42)
model_rf.fit(X_train, y_train)

# 4-3. Time Series Analysis (ARIMA)
# from statsmodels.tsa.arima_model import ARIMA
# model_arima = ARIMA()

print(f'''
      Q. Why not use ARIMA?
      A. Tried to use ARIMA model, but it was not suitable for the dataset
      ''')

# 4-4. Deep Learning Model (LSTM)
# import tensorflow as tf

print(f'''
      Q. Why not use LSTM or RNN?
      A. Couldn't use deep learning models due to the limited CPU, Memory, 
         and GPU resources in cloud environment.
      ''')


# %% 5. MODEL VALIDATION
# 5-1. Model Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 500, 1000, 5000],
    'max_depth': [1, 5, 10, 50, 100, 500, 1000]
}

# Initialize the model
model_rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model_rf, 
                           param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', 
                           n_jobs=4)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print(f'Best parameters: {best_params}')
print(f'Best MSE: {best_score:.2f}')

# Predict using the best model
best_model = grid_search.best_estimator_
y_pred_rf = best_model.predict(X_valid)
mse = mean_squared_error(y_pred_rf, y_valid)
print(f'Validation MSE: {mse:.2f}')

# %% 5-2. Model 
from sklearn.metrics import mean_squared_error

test_period = df_test['date'].between('2024/06/01', '2024/06/30')

x_range = np.arange(1,len(df_test[test_period])+1)
X_test = df_test[test_period][ind_vars]
y_test = df_test[test_period][dep_vars]

y_pred_rf = best_model.predict(X_test)

mse = mean_squared_error(y_pred_rf, y_test)
print(f'MSE: {mse:.2f}')

# 5-3. Model Visualization
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 12))

for i in range(4):
    # 첫 번째 라인 플롯
    axes[i].plot(df_test[test_period]['date'], df_test[test_period][dep_vars[i]],
                 marker='o', markersize=5, label='Actual')
    
    # 두 번째 라인 플롯
    axes[i].plot(df_test[test_period]['date'], y_pred_rf[:, i],
                 marker='o', markersize=5, label='Predicted', linestyle='--')
    
    # 축 제목 및 스타일 설정
    axes[i].set_title(dep_vars[i], fontsize=14)
    axes[i].set_xlabel('Date', fontsize=12)
    axes[i].set_ylabel(f'{dep_vars[i]}\n(억 원)', fontsize=12)
    axes[i].set_xticks(df_test[test_period]['date'])
    axes[i].set_xticklabels(df_test[test_period]['date'].dt.day)
    axes[i].legend(loc='upper left')  # 범례 추가

# 레이아웃 정리
plt.tight_layout()
plt.show()

# 5-4. Model Interpretation
# Feature Importance
feature_importance = best_model.feature_importances_
feature_names = X_train.columns

df_fi = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})

plt.figure(figsize=(12,6))

sns.barplot(data=df_fi, x='importance', y='feature')
plt.title('Feature Importance')
plt.show()

# %%
