# -*- coding: utf-8 -*-
"""Копия блокнота "Clients.ipynb""""""



# Клиенты авиакомпании

Датасет содержит информацию о клиентах некоторой авиакомпании

## Импорт библиотек, константы
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_STATE = 42

DATASET_PATH = "https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/clients.csv"

"""## Загрузка и обзор данных

### Загрузка
"""

# загрузка данных
df = pd.read_csv(DATASET_PATH)

"""### Описание данных

**Целевая переменная**
- `satisfaction`: удовлетворенность клиента полетом, бинарная (*satisfied* или *neutral or dissatisfied*)

**Признаки**
- `Gender` (categorical: _Male_ или _Female_): пол клиента
- `Age` (numeric, int): количество полных лет
- `Customer Type` (categorical: _Loyal Customer_ или _disloyal Customer_): лоялен ли клиент авиакомпании?
- `Type of Travel` (categorical: _Business travel_ или _Personal Travel_): тип поездки
- `Class` (categorical: _Business_ или _Eco_, или _Eco Plus_): класс обслуживания в самолете
- `Flight Distance` (numeric, int): дальность перелета (в милях)
- `Departure Delay in Minutes` (numeric, int): задержка отправления (неотрицательная)
- `Arrival Delay in Minutes` (numeric, int): задержка прибытия (неотрицательная)
- `Inflight wifi service` (categorical, int): оценка клиентом интернета на борту
- `Departure/Arrival time convenient` (categorical, int): оценка клиентом удобство времени прилета и вылета
- `Ease of Online booking` (categorical, int): оценка клиентом удобства онлайн-бронирования
- `Gate location` (categorical, int): оценка клиентом расположения выхода на посадку в аэропорту
- `Food and drink` (categorical, int): оценка клиентом еды и напитков на борту
- `Online boarding` (categorical, int): оценка клиентом выбора места в самолете
- `Seat comfort` (categorical, int): оценка клиентом удобства сиденья
- `Inflight entertainment` (categorical, int): оценка клиентом развлечений на борту
- `On-board service` (categorical, int): оценка клиентом обслуживания на борту
- `Leg room service` (categorical, int): оценка клиентом места в ногах на борту
- `Baggage handling` (categorical, int): оценка клиентом обращения с багажом
- `Checkin service` (categorical, int): оценка клиентом регистрации на рейс
- `Inflight service` (categorical, int): оценка клиентом обслуживания на борту
- `Cleanliness` (categorical, int): оценка клиентом чистоты на борту
"""

# информация от столбцах
df.info()

# случайные три записи из датасета
df.sample(3)

df.describe()

"""В данных есть выбросы."""

df.describe(include='object')

df[['satisfaction']].describe()

sns.histplot(df['satisfaction'])
plt.show()

"""В данных есть пропуски."""

df['satisfaction'].value_counts(dropna=False, normalize=True)

"""Доля пропусков 20%."""

df = df[df.satisfaction != '-']

"""Gender"""

df['Gender'].value_counts(dropna=False, normalize=True)

"""Доля пропусков меньше 1%"""

df = df.dropna()

plt.figure(figsize=(6,4))

sns.countplot(x='Gender', data = df_clean, palette='bright')

Q1 = df['Age'].quantile(q=.25)
Q3 = df['Age'].quantile(q=.75)

#only keep rows in dataframe that have values within 1.5\*IQR of Q1 and Q3
df = df[(df['Age'] > Q1-1.5*(Q3-Q1)) & (df['Age'] < Q3+1.5*(Q3-Q1))& (df['Flight Distance'] > Q1-1.5*(Q3-Q1)) & (df['Flight Distance'] < Q3+1.5*(Q3-Q1))&(df['Departure Delay in Minutes'] > Q1-1.5*(Q3-Q1)) & (df['Departure Delay in Minutes'] < Q3+1.5*(Q3-Q1)) & (df['Arrival Delay in Minutes'] > Q1-1.5*(Q3-Q1)) & (df['Arrival Delay in Minutes'] < Q3+1.5*(Q3-Q1))& (df['Departure/Arrival time convenient'] > Q1-1.5*(Q3-Q1)) & (df['Departure/Arrival time convenient'] < Q3+1.5*(Q3-Q1))]

df = df[(df['Departure/Arrival time convenient'] <= 5)]

df = df[(df['Inflight entertainment'] <= 5) & (df['Checkin service'] <= 5) &(df['Cleanliness'] <= 5)]

df.describe()

corr = df[['Departure/Arrival time convenient','Ease of Online booking', 'Gate location', 'Food and drink','Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service',	'Leg room service',	'Baggage handling',	'Checkin service',	'Inflight service',	'Cleanliness']].corr()

sns.heatmap(corr, cmap="crest")

corr = df[['Food and drink', 'Seat comfort', 'Inflight entertainment',	'Cleanliness']].corr()

sns.heatmap(corr, cmap="crest")

plt.figure(figsize=(6,4))

sns.histplot(x='satisfaction', y='Age', data = df)
plt.title('Satisfaction - Age')
plt.show()

"""Машинное обучение. Предсказание satisfaction"""

X = df.drop(['satisfaction'], axis=1)

y = df['satisfaction']

X.head()

X['Customer Type'] = X['Customer Type'].map({'Loyal Customer' : 1, 'disloyal Customer' : 0})
X['Type of Travel'] = X['Type of Travel'].map({'Business travel' : 1, 'Personal Travel' : 0})
X['Gender'] = X['Gender'].map({'Male' : 1, 'Female' : 0})

X.drop(['Class'], axis=1, inplace=True)

X.head()

y = y.map({'satisfied': 1, 'neutral or dissatisfied': 0}).astype(int)

y[:2]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.shape, X_test.shape

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

pred[:10]

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred)

model.coef_

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)

y_test.value_counts()

from sklearn.metrics import recall_score
recall_score(y_test, pred)

from sklearn.metrics import precision_score

precision_score(y_test, pred)

probs = model.predict_proba(X_test)

probs[:5]

probs[:,1][:5]

classes = probs[:,1] > 0.5

classes[:5]

confusion_matrix(y_test, classes), recall_score(y_test, classes)

classes = probs[:,1] > 0.03

confusion_matrix(y_test, classes), recall_score(y_test, classes)

X_train.head()

from sklearn.preprocessing import MinMaxScaler

ss = MinMaxScaler()
ss.fit(X_train) # вычислить min, max по каждому столбцу

X_train = pd.DataFrame(ss.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(ss.transform(X_test), columns=X_test.columns)

X_train.head()

model = LogisticRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

model.coef_, model.intercept_

importances = pd.DataFrame({'weights': model.coef_[0], 'features': X_train.columns}).sort_values(by='weights')
importances.head()

X = df.drop(['satisfaction'], axis=1)

y = df['satisfaction']

X.dtypes

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical = ['Gender', 'Customer Type', 'Type of Travel']
numeric_features = [col for col in X_train.columns if col not in categorical]

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', handle_unknown="ignore"), categorical),
    ('scaling', MinMaxScaler(), numeric_features)
])

X_train_transformed = column_transformer.fit_transform(X_train)
X_test_transformed = column_transformer.transform(X_test)

X_train_transformed

lst = list(column_transformer.transformers_[0][1].get_feature_names_out())
lst.extend(numeric_features)

X_train_transformed = pd.DataFrame(X_train_transformed, columns=lst)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=lst)

X_train_transformed.head()

model = LogisticRegression()

model.fit(X_train_transformed, y_train)

pred = model.predict_proba(X_test_transformed)[:,1]

classes = pred > 0.5

confusion_matrix(y_test, classes), recall_score(y_test, classes)

classes = pred > 0.01

confusion_matrix(y_test, classes), recall_score(y_test, classes)

importances = pd.DataFrame({'weights': model.coef_[0], 'features': X_train_transformed.columns}).sort_values(by='weights')
importances

import pickle

with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)

#а так модель можно загрузить из файла:
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
