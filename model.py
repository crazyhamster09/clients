import pandas as pd
import numpy as np
import pickle

RANDOM_STATE = 42

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from pickle import dump, load



def split_data(df: pd.DataFrame):
    y = df['satisfaction']
    X = df[["Gender", "Age", "Customer Type", "Type of Travel", "Class", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes", "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"]]

    return X, y


def open_data(path="https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/clients.csv"):
    df = pd.read_csv(path)
    df = df[["satisfaction", "Gender", "Age", "Customer Type", "Type of Travel", "Class", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes", "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"]]

    return df


def preprocess_data(df: pd.DataFrame, test=True):
    df = df[df.satisfaction != '-']
    df.dropna(inplace=True)
    Q1 = df['Age'].quantile(q=.25)
    Q3 = df['Age'].quantile(q=.75)
    df = df[(df['Age'] > Q1-1.5*(Q3-Q1)) & (df['Age'] < Q3+1.5*(Q3-Q1))& (df['Flight Distance'] > Q1-1.5*(Q3-Q1)) & (df['Flight Distance'] < Q3+1.5*(Q3-Q1))&(df['Departure Delay in Minutes'] > Q1-1.5*(Q3-Q1)) & (df['Departure Delay in Minutes'] < Q3+1.5*(Q3-Q1)) & (df['Arrival Delay in Minutes'] > Q1-1.5*(Q3-Q1)) & (df['Arrival Delay in Minutes'] < Q3+1.5*(Q3-Q1))& (df['Departure/Arrival time convenient'] > Q1-1.5*(Q3-Q1)) & (df['Departure/Arrival time convenient'] < Q3+1.5*(Q3-Q1))]
    df = df[(df['Departure/Arrival time convenient'] <= 5)]
    df = df[(df['Inflight entertainment'] <= 5) & (df['Checkin service'] <= 5) &(df['Cleanliness'] <= 5)]

    X['Customer Type'] = X['Customer Type'].map({'Loyal Customer' : 1, 'disloyal Customer' : 0})
    X['Type of Travel'] = X['Type of Travel'].map({'Business travel' : 1, 'Personal Travel' : 0})
    X['Gender'] = X['Gender'].map({'Male' : 1, 'Female' : 0})

    X.drop(['Class'], axis=1, inplace=True)

    y = y.map({'satisfied': 1, 'neutral or dissatisfied': 0}).astype(int)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train.shape, X_test.shape

    ss = MinMaxScaler()
    ss.fit(X_train) # вычислить min, max по каждому столбцу

    X_train = pd.DataFrame(ss.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(ss.transform(X_test), columns=X_test.columns)

    categorical = ['Gender', 'Customer Type', 'Type of Travel']
    numeric_features = [col for col in X_train.columns if col not in categorical]

    column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', handle_unknown="ignore"), categorical),
    ('scaling', MinMaxScaler(), numeric_features)
    ])

    X_train_transformed = column_transformer.fit_transform(X_train)
    X_test_transformed = column_transformer.transform(X_test)

    lst = list(column_transformer.transformers_[0][1].get_feature_names_out())
    lst.extend(numeric_features)

    X_train_transformed = pd.DataFrame(X_train_transformed, columns=lst)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=lst)


def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
    model = LogisticRegression()
    model.fit(X_train_transformed, y_train)
    pred = model.predict(X_test_transformed)

    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    classes = probs[:,1] > 0.03

    confusion_matrix(y_test, classes), recall_score(y_test, classes)

    encode_prediction_proba = {
        0: "Клиент не удовлетворен с вероятностью",
        1: "Клиент удовлетворен с вероятностью"
    }

    encode_prediction = {
        0: "Приносим извинения за доставленные неудобства",
        1: "Спасибо за высокую оценку нашей работы"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)
