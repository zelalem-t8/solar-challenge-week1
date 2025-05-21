from data_loader import load_data
from preprocess import preprocess_data
from model import train_model
from evaluate import evaluate_model

def main():
    df = load_data("data/benin_clean.csv")
    
    target_column = "GHI"
    drop_columns = ["Timestamp"]

    X_train, X_test, y_train, y_test = preprocess_data(df, target_column, drop_columns)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
