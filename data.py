import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 1. Cargar y limpiar datos (como ya tenías)
link = "https://huggingface.co/datasets/hamzas/nba-games/resolve/main/games.csv"
data = pd.read_csv(link)
id_hawks = 1610612737
hawks = data[data['HOME_TEAM_ID'] == id_hawks].copy()
hawks['PTS_home'] = pd.to_numeric(hawks['PTS_home'], errors='coerce')
hawks = hawks.dropna(subset=['PTS_home']).reset_index(drop=True)

# 2. CREAR VARIABLES DE "MEMORIA" (Lags)
# Queremos que el modelo sepa qué pasó en los 3 partidos anteriores
for i in range(1, 4):
    hawks[f'lag_{i}'] = hawks['PTS_home'].shift(i)

# Eliminamos las filas donde no tenemos historia (las primeras 3)
hawks_ml = hawks.dropna().copy()

# Definimos X (los 3 partidos previos) y y (el partido actual)
X = hawks_ml[['lag_1', 'lag_2', 'lag_3']]
y = hawks_ml['PTS_home']

# 3. ENTRENAR UN MODELO MÁS DINÁMICO (Random Forest)
# El Random Forest maneja mejor la volatilidad que la Regresión Lineal
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. PREDICCIÓN "PASO A PASO"
# Para predecir el futuro, necesitamos usar la predicción anterior como dato
ultimos_datos = list(y.tail(3).values)
predicciones_futuras = []

for _ in range(10):
    input_data = np.array(ultimos_datos[-3:]).reshape(1, -1)
    pred = model.predict(input_data)[0]
    predicciones_futuras.append(pred)
    ultimos_datos.append(pred)

# 5. GRAFICAR RESULTADOS RECIENTES (Últimos 50 partidos + 10 predicciones)
plt.figure(figsize=(12, 6))
indices_reales = np.arange(len(y[-50:]))
plt.plot(indices_reales, y[-50:], 'ro-', label='Puntos Reales (Últimos 50)')

indices_pred = np.arange(len(y[-50:]), len(y[-50:]) + 10)
plt.plot(indices_pred, predicciones_futuras, 'b*--', label='Predicción Próximos 10')

plt.axvline(x=len(y[-50:])-1, color='gray', linestyle='--')
plt.title('Predicción Dinámica con Random Forest - Hawks')
plt.legend()
plt.grid(True)
plt.show()

for i, p in enumerate(predicciones_futuras, 1):
    print(f"Partido {i}: {p:.2f} puntos")