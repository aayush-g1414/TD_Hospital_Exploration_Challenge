import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def data_preprocessing(df):
    
    # col_to_keep = ['death', 'age', 'blood', 'reflex', 'bloodchem1', 'bloodchem2', 'psych1', 'glucose']
    # df = df[col_to_keep]

    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)

    X = df

    X = X.drop(X[X['race'] == 0].index)
    X = X.drop(X[X['dnr'] == 0].index)
    X['sex'] = X['sex'].replace(['M', 'Male'], 'male')
    X = X.drop('sex', axis=1)
    # X.head()
    df = X
    return df
    
def split_feature_label(df):
    y = df['death']
    X = df.drop(columns=['death', 'pdeath', 'psych4', 'dose'])

    return y, X

def standardize(X):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import pickle

    # Standardize numeric columns
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric

    # Encode categorical columns using one-hot encoding
    categorical_columns = ['race', 'dnr', 'primary', 'disability', 'income', 'extraprimary', 'cancer']
    X[categorical_columns] = X[categorical_columns].astype(str)

    encoder = OneHotEncoder()
    print(X[categorical_columns].head())
    # Apply transformations to the columns
    ct = ColumnTransformer([('encoder', encoder, categorical_columns)], remainder='passthrough')
    X_transformed = ct.fit_transform(X)

    # Save the scaler and transformer
    scaler_filename = "scaler.pkl"
    encoder_filename = "encoder.pkl"

    with open(scaler_filename, 'wb') as scaler_file, open(encoder_filename, 'wb') as encoder_file:
        pickle.dump(scaler, scaler_file)
        pickle.dump(ct, encoder_file)

    # Return the transformed X
    return X_transformed

def train_model(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)
    from sklearn.linear_model import LogisticRegression
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
    