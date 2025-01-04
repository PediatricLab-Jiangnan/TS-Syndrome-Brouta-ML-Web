import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_prepare_data(file_path):
    """Load and prepare the data."""
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Separate features and target variable
    X = data.drop('Group', axis=1)  # Assuming the target variable column name is 'Group'
    y = data['Group']
    
    return X, y, X.columns

def create_and_train_model(X, y, feature_names):
    """Create and train the model."""
    # Create a scaler for preprocessing
    scaler = StandardScaler()
    
    # Scale the data
    X_scaled = scaler.fit_transform(X)
    
    # Convert the scaled data back to a DataFrame for easier handling and interpretation
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Create and train the Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_scaled, y)
    
    return model, X_scaled

def evaluate_model(model, X_scaled, y):
    """Evaluate the model's performance."""
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    
    # Print evaluation metrics
    print("\nOverall Evaluation Metrics:")
    print(classification_report(y, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nAUC Score:", roc_auc_score(y, y_prob))

def plot_shap_values(model, X_scaled, feature_names):
    """Plot SHAP value visualizations."""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create the figure for SHAP summary plot
    plt.figure(figsize=(12, 8))
    
    # Plot SHAP summary
    shap.summary_plot(
        shap_values, 
        X_scaled,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.show()
    
    # Create the figure for SHAP feature importance bar plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values, 
        X_scaled,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

def main():
    # File path
    file_path = 'your_data.csv'  # Replace with your CSV file path
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data(file_path)
    
    # Create and train the model
    model, X_scaled = create_and_train_model(X, y, feature_names)
    
    # Evaluate the model
    evaluate_model(model, X_scaled, y)
    
    # SHAP value visualization
    print("\nGenerating SHAP value visualizations...")
    plot_shap_values(model, X_scaled, feature_names)
    
    # Save the model and scaler (optional)
    print("\nSaving the model and scaler...")
    joblib.dump(model, 'gbm_model.pkl')
    joblib.dump(StandardScaler(), 'scaler.pkl')  # Save the scaler
    print("Model saved as 'gbm_model.pkl'")
    print("Scaler saved as 'scaler.pkl'")

if __name__ == "__main__":
    main()
