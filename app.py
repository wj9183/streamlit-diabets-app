from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import streamlit as st
import numpy as np


def main():
    model = joblib.load('data/best_model.pkl')
    df = pd.read_csv('data/diabetes.csv')
    st.dataframe(df)

    new_data = np.array([3,66,666,677,8,99,11,22])
    new_data = new_data.reshape(1, -1)

    st.write(model.predict(new_data))
    print(new_data)


if __name__ == '__main__':
    main()