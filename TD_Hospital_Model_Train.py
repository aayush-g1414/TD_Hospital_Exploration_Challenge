import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


def data_preprocessing(df):
    
    # col_to_keep = ['death', 'age', 'blood', 'reflex', 'bloodchem1', 'bloodchem2', 'psych1', 'glucose']
    # df = df[col_to_keep]

    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)

    X = df

    X = X.drop(X[X['race'] == 0].index)
    X = X.drop(X[X['dnr'] == 0].index)
    X = X.drop('dose', axis=1)
    X['sex'] = X['sex'].replace(['M', 'Male'], 'male')
    X = X.drop('sex', axis=1)
    print(X['pdeath'])
    print(X['psych4'])
    # X.head()
    df = X
    return df
    
def split_feature_label(df):
    y = df['death']
    X = df.drop(columns=['death'])
    print(len(X))
    print(len(y))
    return y, X

def standardize(X):
    # Standardize numeric columns
    scaler = StandardScaler()
    
    X_numeric = scaler.fit((X.select_dtypes(include=['float64'])))
    scaler_filename = "scaler.pkl"
    with open(scaler_filename, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    X_numeric = scaler.transform(X.select_dtypes(include=['float64']))
    print(scaler.get_feature_names_out())
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    # print("dose", pd.unique(X['dose']))
    # Encode categorical columns using one-hot encoding
    categorical_columns = ['race', 'dnr', 'primary', 'disability', 'income', 'extraprimary', 'cancer']
    X[categorical_columns] = X[categorical_columns].astype(str)

    encoder = OneHotEncoder()
    # Apply transformations to the columns
    ct = ColumnTransformer([('encoder', encoder, categorical_columns)], remainder='passthrough')
    X_transformed = ct.fit_transform(X)

    # Save the scaler and transformer
    
    encoder_filename = "encoder.pkl"

    with open(encoder_filename, 'wb') as encoder_file:
       
        pickle.dump(ct, encoder_file)

    # Return the transformed X
    return X_transformed

def train_model(X, y):
    # Split data into training and validation
    print(X.shape)
    print(y.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)
    model = LogisticRegression(C=0.1, penalty='l2')
    model.fit(X_train, y_train)
    print(X_test[0])
    
    print(model.score(X_test,y_test))
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)



if __name__ == "__main__":
    data_path = './TD_HOSPITAL_TRAIN.csv'
    df = pd.read_csv(data_path)
    cleaned_data = data_preprocessing(df)
    y, X = split_feature_label(cleaned_data)
    X = standardize(X)
    train_model(X, y)
    