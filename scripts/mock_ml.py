import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. Bikin data bohongan (1 baris data, anggaplah ada 16 kolom fitur kos)
X_dummy = np.zeros((1, 16))
y_dummy = np.array([1500000.0]) # Tebakan harga selalu 1.5 juta

# 2. Latih model Random Forest betulan pakai data bohongan tadi
model = RandomForestRegressor(n_estimators=1, random_state=42)
model.fit(X_dummy, y_dummy)

# 3. Bikin Scaler betulan
scaler = StandardScaler()
scaler.fit(X_dummy)

# 4. Simpan ke dalam file .pkl
with open('models_ml/model_rf.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models_ml/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Dummy Scikit-Learn Model berhasil dibuat! Sekarang sistem pasti kenal.")