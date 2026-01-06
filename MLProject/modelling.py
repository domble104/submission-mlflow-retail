import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import datetime as dt

# --- KONFIGURASI ---
DATA_PATH = "data_clean.csv" 
EXPERIMENT_NAME = "Eksperimen_KMeans_Retail"

# --- FUNGSI BANTUAN (Sama dengan script tuning) ---
def remove_outliers_iqr(df_in, cols):
    clean_df = df_in.copy()
    for col in cols:
        q1, q3 = np.percentile(clean_df[col], [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]
    return clean_df

def main():
    print("=== Memulai Basic Training (Autolog) ===")
    
    # 1. LOAD DATA
    try:
        df = pd.read_csv(DATA_PATH)
        if 'InvoiceDate' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        elif 'Invoice Date' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['Invoice Date'])
    except FileNotFoundError:
        print(f"ERROR: File {DATA_PATH} tidak ditemukan.")
        return

    # 2. RFM CALCULATION
    if 'Total Amount' not in df.columns:
        df['Total Amount'] = df['Quantity'] * df['Price']
    
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    cust_col = 'Customer ID' if 'Customer ID' in df.columns else 'CustomerID'
    
    rfm = df.groupby(cust_col).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'Total Amount': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Total Amount': 'Monetary'})

    # 3. PREPROCESSING
    # Hapus Outlier
    rfm_clean = remove_outliers_iqr(rfm, ["Recency", "Frequency", "Monetary"])
    
    # Transformasi Log & Scaling
    rfm_features = rfm_clean.copy()
    rfm_features["Frequency_log"] = np.log1p(rfm_features["Frequency"])
    rfm_features["Monetary_log"] = np.log1p(rfm_features["Monetary"])
    
    X = rfm_features[["Recency", "Frequency_log", "Monetary_log"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. TRAINING DENGAN AUTOLOG (Syarat Basic)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # KUNCI: Aktifkan Autolog sebelum training
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Basic_Run_Autolog"):
        print("-> Melatih model k=3 dengan Autolog...")
        
        # Kita pilih k=3 sebagai contoh model 'basic'
        model = KMeans(n_clusters=3, init='k-means++', random_state=42)
        model.fit(X_scaled)
        
        # TIDAK PERLU manual logging (log_param, log_metric, log_model)
        # MLflow akan otomatis menangkapnya karena perintah autolog() di atas.
        
        print("-> Training selesai! Metrik dan parameter sudah tersimpan otomatis.")

if __name__ == "__main__":
    main()