import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_house_price_dashboard():
    try:
        # Load Housing.csv dataset
        data = pd.read_csv('Housing.csv')
        print("Dataset loaded successfully!")
        
        # Initial data inspection
        print(f"Dataset shape: {data.shape}")
        print(f"Missing values: {data.isnull().sum().sum()}")
        
        # Handle missing values
        data = data.dropna()
        
        # Convert categorical variables to numerical
        le = LabelEncoder()
        for col in data.select_dtypes(include=['object']).columns:
            if col != 'price':
                data[col] = le.fit_transform(data[col])
        
        # Feature selection
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        target_col = 'price'
        feature_cols = [col for col in numerical_cols if col != target_col]
        
        X = data[feature_cols]
        y = data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(abs(y_test - y_pred))
        rmse = np.sqrt(mse)
        
        # Create essential prediction dashboard
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle('House Price Prediction Dashboard', fontsize=18, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        plt.subplot(2, 3, 1)
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        plt.subplot(2, 3, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Prices')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # 3. Feature importance (coefficients)
        plt.subplot(2, 3, 3)
        feature_names = X.columns
        coefficients = model.coef_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False).head(8)
        
        plt.barh(range(len(feature_importance)), feature_importance['coefficient'], color='orange')
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Top Features')
        plt.gca().invert_yaxis()
        
        # 4. Prediction error distribution
        plt.subplot(2, 3, 4)
        errors = abs(y_test - y_pred)
        plt.hist(errors, bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Errors')
        plt.grid(True, alpha=0.3)
        
        # 5. Model performance metrics
        plt.subplot(2, 3, 5)
        plt.axis('off')
        performance_text = f"""MODEL PERFORMANCE

R² Score: {r2:.3f}
RMSE: {rmse:,.0f}
MAE: {mae:,.0f}

Samples: {len(X_test)}
Features: {len(feature_cols)}

Accuracy: {r2*100:.1f}%"""
        
        plt.text(0.1, 0.5, performance_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # 6. Feature correlation with price
        plt.subplot(2, 3, 6)
        correlations = data[feature_cols + [target_col]].corr()[target_col].drop(target_col)
        correlations = correlations.sort_values(key=abs, ascending=False).head(8)
        
        plt.barh(range(len(correlations)), correlations.values, color='cyan')
        plt.yticks(range(len(correlations)), correlations.index)
        plt.xlabel('Correlation with Price')
        plt.title('Feature Correlations')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('house_price_prediction_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"R-squared Score: {r2:.3f}")
        print(f"Mean Squared Error: {mse:.0f}")
        print(f"Root Mean Squared Error: {rmse:.0f}")
        print(f"Mean Absolute Error: {mae:.0f}")
        print(f"\nDashboard saved as: house_price_prediction_dashboard.png")
        
        return True
        
    except FileNotFoundError:
        print("Error: Housing.csv file not found. Please ensure the file exists in the current directory.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("HOUSE PRICE PREDICTION SYSTEM")
    print("=" * 50)
    success = create_house_price_dashboard()
    if success:
        print("=" * 50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
    else:
        print("ANALYSIS FAILED!")