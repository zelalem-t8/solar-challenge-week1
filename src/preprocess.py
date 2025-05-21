from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(df, target_column, drop_cols=[]):
    df = df.dropna()  # Simple missing value handling

    X = df.drop([target_column] + drop_cols, axis=1)
    y = df[target_column]

    # Encode categorical features
    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
