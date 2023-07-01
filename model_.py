import pandas as pd
import numpy as np
import pickle

RANDOM_STATE = 42

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pickle import dump, load



def open_data(path="https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/clients.csv"):
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)
    Q1 = df['Age'].quantile(q=.25)
    Q3 = df['Age'].quantile(q=.75)
    df_preprocessed = df[(df['Age'] > Q1-1.5*(Q3-Q1)) & (df['Age'] < Q3+1.5*(Q3-Q1))& (df['Flight Distance'] > Q1-1.5*(Q3-Q1)) & (df['Flight Distance'] < Q3+1.5*(Q3-Q1))&(df['Departure Delay in Minutes'] > Q1-1.5*(Q3-Q1)) & (df['Departure Delay in Minutes'] < Q3+1.5*(Q3-Q1)) & (df['Arrival Delay in Minutes'] > Q1-1.5*(Q3-Q1)) & (df['Arrival Delay in Minutes'] < Q3+1.5*(Q3-Q1))& (df['Departure/Arrival time convenient'] > Q1-1.5*(Q3-Q1)) & (df['Departure/Arrival time convenient'] < Q3+1.5*(Q3-Q1))]
    df_preprocessed = df[(df['Departure/Arrival time convenient'] <= 5)]
    df_preprocessed = df[(df['Inflight entertainment'] <= 5) & (df['Checkin service'] <= 5) &(df['Cleanliness'] <= 5)]
    return df_preprocessed


def split_data(df: pd.DataFrame):
    y = df['satisfaction']
    X = df[["Gender", "Age", "Customer Type", "Type of Travel", "Class", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes", "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"]]

    return X, y

    train_examples = df['x_train']
    train_labels = df['y_train']
    test_examples = df['x_test']
    test_labels = df['y_test']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    

    to_encode = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for col in to_encode:
        dummy = pd.get_dummies(X_df[col], prefix=col)
        X_df = pd.concat([X_df, dummy], axis=1)
        X_df.drop(col, axis=1, inplace=True)

    if test:
        return X_df, y_df
    else:
        return X_df




def fit_and_save_model(X_df, y_df):
    model = LogisticRegression()
    model.fit(X_df, y_df)
    pred = model.predict(X_df)

    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open('model.pickle', "wb") as file:
        dump(model, file)

    print(f"Model was saved to {'./data/model.pickle'}")


def load_model_and_predict(df):
    with open('./data/model.pickle', "rb") as file:
        model = load(file)

    prediction = model.predict(df)
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)
    # prediction_proba = np.squeeze(prediction_proba)


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
