# DS-ML-Showcase
# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
# model_evaluation.py
from model_training import train_model
from sklearn.metrics import classification_report
from data_preprocessing import load_and_preprocess_data

def evaluate_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = train_model()
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
