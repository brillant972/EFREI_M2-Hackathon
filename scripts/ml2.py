import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_flood_target(df, rr_threshold=50, rr_7d_threshold=150):
    df = df.copy()
    
    df['flood'] = (
        (df['RR'] >= rr_threshold) | 
        (df['RR_7d'] >= rr_7d_threshold)
    ).astype(int)
    
    return df

def add_temporal_features(df):
    df = df.copy()
    
    # Ensure date column exists and is datetime
    if 'date' not in df.columns:
        print("Warning: 'date' column not found. Creating dummy dates.")
        df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Basic temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['weekday'] = df['date'].dt.weekday
    
    # Seasonal categories
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    df['season'] = df['month'].apply(get_season)
    
    # Cyclical encoding for temporal features
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    return df

def analyze_seasonal_flood_patterns(df):
    print("------------------------------------------------------------------------------------------------------")
    print("SEASONAL FLOOD PATTERNS")
    print("OVERALL STATS")
    # Overall statistics
    total_obs = len(df)
    total_floods = df['flood'].sum()
    flood_rate = df['flood'].mean() * 100
    
    print(f"Total observations: {total_obs}")
    print(f"Total floods: {total_floods}")
    print(f"Overall flood rate: {flood_rate:.2f}%\n")
    
    # Monthly analysis
    monthly_stats = df.groupby('month')['flood'].agg([
        'count', 'sum', 'mean'
    ]).round(3)
    monthly_stats.columns = ['total_days', 'flood_days', 'flood_probability']
    monthly_stats['flood_probability'] *= 100
    monthly_stats['month_name'] = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    monthly_stats = monthly_stats[['month_name', 'total_days', 'flood_days', 'flood_probability']]
    
    print("------------------------------------------------------------------------------------------------------")
    print("MONTHLY STATS")
    print(monthly_stats.to_string())
    print()
    
    # Weekly analysis
    weekly_stats = df.groupby('week_of_year')['flood'].agg([
        'count', 'sum', 'mean'
    ]).round(3)
    weekly_stats.columns = ['total_days', 'flood_days', 'flood_probability']
    weekly_stats['flood_probability'] *= 100
    
    # Find top risk weeks
    top_risk_weeks = weekly_stats.nlargest(10, 'flood_probability')
    
    print("------------------------------------------------------------------------------------------------------")
    print("TOP 10 HIGHEST RISK WEEKS")
    print("Week | Days | Floods | Probability")
    print("-" * 35)
    for week, row in top_risk_weeks.iterrows():
        print(f"{week:4d} | {row['total_days']:4.0f} | {row['flood_days']:6.0f} | {row['flood_probability']:8.1f}%")
    print()
    
    # Seasonal analysis
    seasonal_stats = df.groupby('season')['flood'].agg([
        'count', 'sum', 'mean'
    ]).round(3)
    seasonal_stats.columns = ['total_days', 'flood_days', 'flood_probability']
    seasonal_stats['flood_probability'] *= 100
    
    print("------------------------------------------------------------------------------------------------------")
    print("SEASONAL STATS")
    print(seasonal_stats.to_string())
    print()
    
    # Quarterly analysis
    quarterly_stats = df.groupby('quarter')['flood'].agg([
        'count', 'sum', 'mean'
    ]).round(3)
    quarterly_stats.columns = ['total_days', 'flood_days', 'flood_probability']
    quarterly_stats['flood_probability'] *= 100
    quarterly_stats['quarter_name'] = ['Q1', 'Q2', 'Q3', 'Q4']
    quarterly_stats = quarterly_stats[['quarter_name', 'total_days', 'flood_days', 'flood_probability']]
    
    print("------------------------------------------------------------------------------------------------------")
    print("QUARTERLY STATS")
    print(quarterly_stats.to_string())
    print()
    
    return {
        'monthly': monthly_stats,
        'weekly': weekly_stats,
        'seasonal': seasonal_stats,
        'quarterly': quarterly_stats,
        'top_risk_weeks': top_risk_weeks
    }

def prepare_seasonal_features(df):
    df = df.copy()
    
    base_features = ['RR', 'RR_7d', 'RR_14d', 'RR_30d', 'API', 
                    'TM', 'TM_7d', 'TM_30d', 'PMER', 'FFM', 'ALTI']
    
    temporal_features = ['month', 'day_of_year', 'week_of_year', 'quarter',
                        'sin_month', 'cos_month', 'sin_doy', 'cos_doy', 
                        'sin_week', 'cos_week']
    
    all_features = base_features + temporal_features
    available_features = [f for f in all_features if f in df.columns]
    
    if len(available_features) != len(all_features):
        missing = set(all_features) - set(available_features)
        print(f"Missing features: {missing}")
    
    X = df[available_features].fillna(df[available_features].median())
    
    return X, available_features

def train_seasonal_model(X, y, model_type='rf'):
    print(f"\n=== TRAINING SEASONAL MODEL ({model_type.upper()}) ===")
    
    # Temporal split (use last 20% of data as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training floods: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"Test floods: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        scaler = None
    
    if model_type == 'rf':
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc_score:.3f}")
    
    # Feature importance for Random Forest
    if model_type == 'rf':
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("------------------------------------------------------------------------------------------------------")
        print("\nTOP 10 FEATURE IMPORTANCES")
        print(feature_importance.head(10).to_string(index=False))
    
    return model, scaler, X_test, y_test, y_pred_proba

def predict_seasonal_risk(model, scaler, df_features, temporal_stats):
    print("------------------------------------------------------------------------------------------------------")
    print("\nSEASONAL RISK PREDICTIONS")
    
    # Prepare features
    if scaler is not None:
        df_scaled = scaler.transform(df_features)
        flood_probs = model.predict_proba(df_scaled)[:, 1]
    else:
        flood_probs = model.predict_proba(df_features)[:, 1]
    
    df_with_pred = df_features.copy()
    df_with_pred['flood_probability'] = flood_probs
    
    # Weekly predictions
    weekly_pred = df_with_pred.groupby('week_of_year')['flood_probability'].mean()
    top_risk_weeks_pred = weekly_pred.nlargest(10)
    
    print("------------------------------------------------------------------------------------------------------")
    print("TOP 10 PREDICTED HIGH-RISK WEEKS")
    print("Week | Avg Flood Probability")
    print("-" * 30)
    for week, prob in top_risk_weeks_pred.items():
        print(f"{week:4d} | {prob*100:16.1f}%")
    print()
    
    # Monthly predictions
    monthly_pred = df_with_pred.groupby('month')['flood_probability'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print("------------------------------------------------------------------------------------------------------")
    print("MONTHLY FLOOD RISK PREDICTIONS")
    print("Month | Avg Flood Probability")
    print("-" * 30)
    for month, prob in monthly_pred.items():
        print(f"{month_names[month-1]:5s} | {prob*100:16.1f}%")
    print()
    
    # Seasonal predictions
    seasonal_mapping = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
                       5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
                       9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
    
    df_with_pred['season'] = df_with_pred['month'].map(seasonal_mapping)
    seasonal_pred = df_with_pred.groupby('season')['flood_probability'].mean()
    
    print("------------------------------------------------------------------------------------------------------")
    print("SEASONAL FLOOD RISK PREDICTIONS")
    print("Season | Avg Flood Probability")
    print("-" * 30)
    for season, prob in seasonal_pred.items():
        print(f"{season:6s} | {prob*100:16.1f}%")
    print()
    
    return {
        'weekly': weekly_pred,
        'monthly': monthly_pred,
        'seasonal': seasonal_pred,
        'top_risk_weeks': top_risk_weeks_pred
    }

def seasonal_flood_pipeline(df, rr_threshold=50, rr_7d_threshold=150, model_type='rf'):
    
    df_with_target = create_flood_target(df, rr_threshold, rr_7d_threshold)
    
    df_temporal = add_temporal_features(df_with_target)

    temporal_stats = analyze_seasonal_flood_patterns(df_temporal)

    X, feature_names = prepare_seasonal_features(df_temporal)
    y = df_temporal['flood']
    
    print(f"Features used: {len(feature_names)}")
    print(f"Data shape: {X.shape}")

    model, scaler, X_test, y_test, y_pred_proba = train_seasonal_model(X, y, model_type)

    predictions = predict_seasonal_risk(model, scaler, X, temporal_stats)
    
    print("------------------------------------------------------------------------------------------------------")
    print("\nPIPELINE COMPLETED SUCCESSFULLY")
    
    return {
        'model': model,
        'scaler': scaler,
        'features': feature_names,
        'temporal_stats': temporal_stats,
        'predictions': predictions,
        'X': X,
        'y': y
    }

def plot_seasonal_patterns(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    monthly_hist = results['temporal_stats']['monthly']['flood_probability']
    monthly_pred = results['predictions']['monthly'] * 100
    
    ax1 = axes[0, 0]
    x = range(1, 13)
    ax1.bar([i-0.2 for i in x], monthly_hist, width=0.4, label='Historical', alpha=0.7)
    ax1.bar([i+0.2 for i in x], monthly_pred, width=0.4, label='Predicted', alpha=0.7)
    ax1.set_title('Monthly Flood Probability')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Flood Probability (%)')
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    
    # Weekly patterns
    weekly_pred = results['predictions']['weekly'] * 100
    ax2 = axes[0, 1]
    ax2.plot(weekly_pred.index, weekly_pred.values)
    ax2.set_title('Weekly Flood Probability Throughout Year')
    ax2.set_xlabel('Week of Year')
    ax2.set_ylabel('Flood Probability (%)')
    ax2.grid(True, alpha=0.3)
    
    # Seasonal comparison
    seasonal_hist = results['temporal_stats']['seasonal']['flood_probability']
    seasonal_pred = results['predictions']['seasonal'] * 100
    
    ax3 = axes[1, 0]
    seasons = ['Autumn', 'Spring', 'Summer', 'Winter']
    hist_vals = [seasonal_hist[s] for s in seasons]
    pred_vals = [seasonal_pred[s] for s in seasons]
    
    x_pos = range(len(seasons))
    ax3.bar([i-0.2 for i in x_pos], hist_vals, width=0.4, label='Historical', alpha=0.7)
    ax3.bar([i+0.2 for i in x_pos], pred_vals, width=0.4, label='Predicted', alpha=0.7)
    ax3.set_title('Seasonal Flood Probability')
    ax3.set_ylabel('Flood Probability (%)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(seasons)
    ax3.legend()
    
    # Top risk weeks
    top_weeks = results['predictions']['top_risk_weeks'] * 100
    ax4 = axes[1, 1]
    ax4.bar(range(len(top_weeks)), top_weeks.values)
    ax4.set_title('Top 10 Highest Risk Weeks')
    ax4.set_xlabel('Week Rank')
    ax4.set_ylabel('Flood Probability (%)')
    ax4.set_xticks(range(len(top_weeks)))
    ax4.set_xticklabels([f"W{w}" for w in top_weeks.index], rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample_df = pd.read_parquet("data/silver/time_series/meteo_clean.parquet")
    
    results = seasonal_flood_pipeline(sample_df, rr_threshold=20, rr_7d_threshold=80)
    
    plot_seasonal_patterns(results)