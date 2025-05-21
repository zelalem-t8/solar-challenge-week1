from sklearn.ensemble import GradientBoostingRegressor  # or Classifier depending on your task

def train_model(X_train, y_train):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model
