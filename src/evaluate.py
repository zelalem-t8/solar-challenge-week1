from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # --- Plot 1: Predicted vs Actual ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Actual GHI")
    plt.ylabel("Predicted GHI")
    plt.title("Predicted vs Actual GHI")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Residual Plot ---
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color='purple')
    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
