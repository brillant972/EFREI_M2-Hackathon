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

def diagnose_perfect_model(df, model, X, y):
    """
    Diagnostique pourquoi le modèle est "trop parfait"
    """
    print("=== DIAGNOSTIC DU MODÈLE PARFAIT ===")
    
    # 1. Vérifier la corrélation entre target et features
    df_corr = df.copy()
    df_corr['flood'] = y
    
    # Calculer les corrélations avec la target
    correlations = df_corr[X.columns.tolist() + ['flood']].corr()['flood'].sort_values(ascending=False)
    
    print("Corrélations avec la variable 'flood' :")
    print(correlations[:-1])  # Exclure la corrélation avec elle-même
    print()
    
    # 2. Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance (Random Forest) :")
        print(importance_df)
        print()
        
        # Les 3 features les plus importantes
        top_features = importance_df.head(3)['feature'].tolist()
        
        print(f"Top 3 features : {top_features}")
        
        # Analyser la distribution de ces features pour les cas d'inondation
        for feature in top_features[:2]:  # Analyser les 2 plus importantes
            print(f"\n--- Analyse de {feature} ---")
            flood_values = df_corr[df_corr['flood'] == 1][feature]
            no_flood_values = df_corr[df_corr['flood'] == 0][feature]
            
            print(f"Inondations - {feature}: min={flood_values.min():.1f}, max={flood_values.max():.1f}, mean={flood_values.mean():.1f}")
            print(f"Pas d'inondation - {feature}: min={no_flood_values.min():.1f}, max={no_flood_values.max():.1f}, mean={no_flood_values.mean():.1f}")
    
    return correlations, importance_df if hasattr(model, 'feature_importances_') else None

def create_realistic_target(df, percentile_threshold=99):
    """
    Crée une variable cible plus réaliste basée sur des percentiles
    """
    print(f"=== CRÉATION D'UNE TARGET PLUS RÉALISTE ===")
    
    # Calculer les percentiles par station pour tenir compte des spécificités locales
    if 'NUM_POSTE' in df.columns:
        station_thresholds = df.groupby('NUM_POSTE')['RR'].quantile(percentile_threshold/100)
        df = df.copy()
        df['station_threshold'] = df['NUM_POSTE'].map(station_thresholds)
        
        # Inondation si pluie > percentile 99 de la station OU cumul 7j très élevé
        df['flood_realistic'] = (
            (df['RR'] > df['station_threshold']) |
            (df['RR_7d'] > df.groupby('NUM_POSTE')['RR_7d'].transform(lambda x: x.quantile(0.98)))
        ).astype(int)
    else:
        # Version globale
        rr_threshold = df['RR'].quantile(percentile_threshold/100)
        rr_7d_threshold = df['RR_7d'].quantile(0.98)
        
        df = df.copy()
        df['flood_realistic'] = (
            (df['RR'] >= rr_threshold) | 
            (df['RR_7d'] >= rr_7d_threshold)
        ).astype(int)
    
    print(f"Nouveau taux d'inondations : {df['flood_realistic'].mean()*100:.2f}%")
    print(f"Nombre d'inondations : {df['flood_realistic'].sum()}")
    
    return df

def temporal_validation(X, y, df):
    """
    Validation temporelle pour éviter le data leakage
    """
    print("\n=== VALIDATION TEMPORELLE ===")
    
    if 'date' not in df.columns:
        print("Pas de colonne 'date' disponible pour la validation temporelle")
        return None
    
    # Tri par date
    df_sorted = df.copy()
    df_sorted['target'] = y
    df_sorted = df_sorted.sort_values('date')
    
    # Split temporel : 80% pour train, 20% pour test
    split_date = df_sorted['date'].quantile(0.8)
    
    train_mask = df_sorted['date'] < split_date
    test_mask = df_sorted['date'] >= split_date
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Période d'entraînement : {df_sorted[train_mask]['date'].min()} à {df_sorted[train_mask]['date'].max()}")
    print(f"Période de test : {df_sorted[test_mask]['date'].min()} à {df_sorted[test_mask]['date'].max()}")
    print(f"Taille train : {len(X_train)} ({y_train.mean()*100:.2f}% inondations)")
    print(f"Taille test : {len(X_test)} ({y_test.mean()*100:.2f}% inondations)")
    
    # Entraînement avec validation temporelle
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=100,  # Régularisation
        min_samples_leaf=50,    # Régularisation
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nRésultats avec validation temporelle :")
    print(f"AUC-ROC : {auc_score:.3f}")
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test, y_pred_proba, auc_score

def advanced_features(df):
    """
    Crée des features plus avancées pour améliorer le modèle
    """
    print("\n=== CRÉATION DE FEATURES AVANCÉES ===")
    
    df = df.copy()
    
    # 1. Intensité pluviométrique
    df['RR_intensity'] = df['RR'] / (df['RR_7d'].clip(lower=1))  # Éviter division par 0
    
    # 2. Gradient de pluie (changement rapide)
    if 'NUM_POSTE' in df.columns:
        df['RR_diff'] = df.groupby('NUM_POSTE')['RR'].diff().fillna(0)
        df['RR_7d_diff'] = df.groupby('NUM_POSTE')['RR_7d'].diff().fillna(0)
    else:
        df['RR_diff'] = df['RR'].diff().fillna(0)
        df['RR_7d_diff'] = df['RR_7d'].diff().fillna(0)
    
    # 3. Features saisonnières renforcées
    if 'date' in df.columns:
        df['month'] = df['date'].dt.month
        # Encodage des mois à risque (mai-septembre)
        df['high_risk_season'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    
    # 4. Interaction entre température et pluie (fonte de neige + pluie)
    df['temp_rain_interaction'] = df['TM'] * df['RR']
    
    # 5. Saturation des sols + pluie intense
    df['API_RR_interaction'] = df['API'] * df['RR']
    
    # 6. Features climatologiques
    if 'NUM_POSTE' in df.columns:
        # Écart à la normale par station
        station_normals = df.groupby(['NUM_POSTE', df['date'].dt.month])['RR'].transform('mean')
        df['RR_anomaly'] = df['RR'] - station_normals
    
    new_features = ['RR_intensity', 'RR_diff', 'RR_7d_diff', 'high_risk_season', 
                   'temp_rain_interaction', 'API_RR_interaction']
    
    if 'RR_anomaly' in df.columns:
        new_features.append('RR_anomaly')
    
    print(f"Nouvelles features créées : {new_features}")
    
    return df, new_features

def improved_model_pipeline(df, use_realistic_target=True):
    """
    Pipeline amélioré avec validation temporelle et features avancées
    """
    print("=== PIPELINE AMÉLIORÉ ===")
    
    # 1. Créer target réaliste
    if use_realistic_target:
        df = create_realistic_target(df, percentile_threshold=99.5)
        target_col = 'flood_realistic'
    else:
        target_col = 'flood'
    
    # 2. Créer features avancées
    df, new_features = advanced_features(df)
    
    # 3. Features pour le modèle
    base_features = ['RR', 'RR_7d', 'RR_14d', 'RR_30d', 'API', 
                    'TM', 'TM_7d', 'TM_30d', 'FFM', 'sin_doy', 'cos_doy', 'ALTI']
    
    all_features = base_features + new_features
    available_features = [f for f in all_features if f in df.columns]
    
    print(f"Features utilisées : {available_features}")
    
    # 4. Préparer les données
    X = df[available_features].fillna(df[available_features].median())
    y = df[target_col]
    
    # 5. Validation temporelle
    results = temporal_validation(X, y, df)
    
    if results:
        model, X_test, y_test, y_pred_proba, auc_score = results
        
        # Feature importance du modèle amélioré
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Features Importantes (Modèle Amélioré) :")
            print(importance_df.head(10))
        
        return model, X, y, y_pred_proba, auc_score
    
    return None

# FONCTION PRINCIPALE POUR DIAGNOSTIQUER ET AMÉLIORER
def diagnose_and_improve_model(df, original_model, original_X, original_y):
    """
    Fonction principale pour diagnostiquer le modèle parfait et l'améliorer
    """
    print("=" * 60)
    print("DIAGNOSTIC ET AMÉLIORATION DU MODÈLE")
    print("=" * 60)
    
    # 1. Diagnostiquer le modèle original
    correlations, importance = diagnose_perfect_model(df, original_model, original_X, original_y)
    
    # 2. Créer un modèle amélioré
    print("\n" + "="*50)
    improved_results = improved_model_pipeline(df, use_realistic_target=True)
    
    if improved_results:
        model, X, y, y_pred_proba, auc_score = improved_results
        print(f"\n🎯 RÉSULTAT FINAL :")
        print(f"AUC-ROC du modèle amélioré : {auc_score:.3f}")
        print(f"Plus réaliste que le modèle original (AUC=1.000)")
        
        return model, X, y
    else:
        print("Impossible de créer le modèle amélioré")
        return None

if __name__ == "__main__":
    sample_df = pd.read_parquet("data/silver/time_series/meteo_clean.parquet")
    
    model, X, y, threshold = main_pipeline(sample_df, rr_threshold=20, rr_7d_threshold=80)

    sample_df['date'] = pd.to_datetime(sample_df['date'])

    # Diagnostiquer le modèle actuel
    improved_model, X_new, y_new = diagnose_and_improve_model(
        sample_df, model, X, y
    )
    
    print(f"\n=== MODÈLE ENTRAÎNÉ AVEC SUCCÈS ===")
    print(f"Seuil de décision optimal : {threshold:.3f}")