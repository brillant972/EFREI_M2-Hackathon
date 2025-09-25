import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def create_flood_target(df, rr_threshold=50, rr_7d_threshold=150):
    df = df.copy()

    df['flood'] = (
        (df['RR'] >= rr_threshold) | 
        (df['RR_7d'] >= rr_7d_threshold)
    ).astype(int)
    
    return df

def prepare_features(df):
    df = df.copy()
    
    features = ['RR', 'RR_7d', 'RR_14d', 'RR_30d', 'API', 
                'TM', 'TM_7d', 'TM_30d', 'PMER', 'FFM',
                'sin_doy', 'cos_doy', 'ALTI']
    
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) != len(features):
        missing = set(features) - set(available_features)
        print(f"Features manquantes : {missing}")
        features = available_features
    
    X = df[features].fillna(df[features].median())
    
    return X, features

def analyze_flood_distribution(df):
    print("=== FLOOD DISTRIBUTION ===")
    print(f"Nombre total d'observations : {len(df)}")
    print(f"Nombre d'inondations : {df['flood'].sum()}")
    print(f"Pourcentage d'inondations : {df['flood'].mean()*100:.2f}%")
    print()
    
    monthly_floods = df.groupby(df['date'].dt.month)['flood'].agg(['count', 'sum', 'mean'])
    monthly_floods.columns = ['total_obs', 'flood_count', 'flood_rate']
    monthly_floods['flood_rate'] = monthly_floods['flood_rate'] * 100
    
    print("Distribution mensuelle des inondations :")
    print(monthly_floods)
    
    return monthly_floods

def train_model(X, y, model_type='rf'):
    print(f"\n=== ENTRAÎNEMENT DU MODÈLE ({model_type.upper()}) ===")
    
    # Split temporel si on a des dates, sinon split aléatoire
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
    elif model_type == 'lr':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    

    print(f"Taille train : {len(X_train)}")
    print(f"Taille test : {len(X_test)}")
    print(f"Inondations dans test : {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    print()
    
    print("MÉTRIQUES :")
    print(f"AUC-ROC : {roc_auc_score(y_test, y_pred_proba):.3f}")
    print()
    print("Classification Report :")
    print(classification_report(y_test, y_pred))
    
    print("Matrice de confusion :")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    if model_type == 'rf':
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    return model, X_test, y_test, y_pred_proba

def find_optimal_threshold(y_true, y_pred_proba):
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nSeuil optimal : {optimal_threshold:.3f}")
    print(f"F1-score optimal : {f1_scores[optimal_idx]:.3f}")
    print(f"Precision : {precision[optimal_idx]:.3f}")
    print(f"Recall : {recall[optimal_idx]:.3f}")
    
    return optimal_threshold

def main_pipeline(df, rr_threshold=50, rr_7d_threshold=150):
    print("=== PIPELINE DE PRÉDICTION D'INONDATIONS ===")
    
    df_with_target = create_flood_target(df, rr_threshold, rr_7d_threshold)
    
    monthly_stats = analyze_flood_distribution(df_with_target)
    
    X, feature_names = prepare_features(df_with_target)
    y = df_with_target['flood']
    
    print(f"\nFeatures utilisées : {feature_names}")
    print(f"Forme des données : {X.shape}")
    
    model, X_test, y_test, y_pred_proba = train_model(X, y, model_type='rf')

    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
    
    return model, X, y, optimal_threshold

def create_sample_data():
    np.random.seed(42)
    n_samples = 10000
    
    dates = pd.date_range('2018-01-01', periods=n_samples, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'NUM_POSTE': np.random.choice(['12345', '23456', '34567'], n_samples),
        'RR': np.random.exponential(2, n_samples), 
        'ALTI': np.random.uniform(0, 1000, n_samples),
        'TM': 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 3, n_samples),
        'PMER': 1013 + np.random.normal(0, 10, n_samples),
        'FFM': np.random.exponential(3, n_samples),
        'API': np.random.uniform(0, 100, n_samples),
    })
    
    df = df.sort_values(['NUM_POSTE', 'date'])
    df['RR_7d'] = df.groupby('NUM_POSTE')['RR'].rolling(7, min_periods=1).sum().values
    df['RR_14d'] = df.groupby('NUM_POSTE')['RR'].rolling(14, min_periods=1).sum().values
    df['RR_30d'] = df.groupby('NUM_POSTE')['RR'].rolling(30, min_periods=1).sum().values
    df['TM_7d'] = df.groupby('NUM_POSTE')['TM'].rolling(7, min_periods=1).mean().values
    df['TM_30d'] = df.groupby('NUM_POSTE')['TM'].rolling(30, min_periods=1).mean().values

    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

if __name__ == "__main__":
    sample_df = pd.read_parquet("data/silver/time_series/meteo_clean.parquet")
    
    model, X, y, threshold = main_pipeline(sample_df, rr_threshold=20, rr_7d_threshold=80)
    
    print(f"\n=== MODÈLE ENTRAÎNÉ AVEC SUCCÈS ===")
    print(f"Seuil de décision optimal : {threshold:.3f}")