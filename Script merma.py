# IMPLEMENTACIÓN DE ANÁLISIS PREDICTIVO COMPLETO
# Utilizamos train.csv disponible en https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting (Es necesario 
# registrarse en la página de kaggle). Luego incorporar el archivo en el directorio de trabajo con python.
# Las librerias necesarias están en el archivo requirements.txt

# PASO 1: IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import datetime as dt

# ---- RANDOM FOREST ----
# ÁRBOLES DE DECISIÓN
#Ajuste la cantidad de arboles de desición que necesite (n_estimators). Mientras más mayor es el tiempo de computo, pero mayor presición (hasta cierto punto).
rf_a_decision= 150

# SEMILLA
#Ajuste la cantidad de "semilla"  fijar la "aleatoriedad" para que los resultados sean consistentes y comparables entre ejecuciones. (random_state)
rf_a_random= 50

# --- XGBoost ---
# ÁRBOLES DE DECISIÓN
#Ajuste la cantidad de arboles de desición que necesite (n_estimators). Mientras más mayor es el tiempo de computo, pero mayor presición (hasta cierto punto).
xgb_a_decision= 500

# SEMILLA
#Ajuste la cantidad de "semilla"  fijar la "aleatoriedad" para que los resultados sean consistentes y comparables entre ejecuciones.(random_state)
xgb_a_random= 100

print("*IMPLEMENTACIÓN DE MODELOS PREDICTIVOS. CASO PREDICTIVO DE VENTAS*")

# PASO 2: CARGA Y PREPARACIÓN DE DATOS
# Cargar el dataset (Para mermas es posible transformar a CSV o modificar script para conectarse a la base de datos. Queda a elecccion del grupo)
data = pd.read_csv('mermas_utf8_clean.csv')

# Convertir fechas a formato datetime con formato día/mes/año
data['fecha'] = pd.to_datetime(data['fecha'], format='%Y-%m-%d', dayfirst=True)

q99 = data['merma_unidad_p'].quantile(0.99)
data = data[data['merma_unidad_p'] <= q99]

# Crear nuevas características para las fechas
data['anio'] = data['fecha'].dt.year
data['mes'] = data['fecha'].dt.month
data['dia'] = data['fecha'].dt.day
data['dia_semana'] = data['fecha'].dt.dayofweek

data = data.drop(columns=["fecha"])

# PASO 3: SELECCIÓN DE CARACTERÍSTICAS
# Características para predecir ventas. Estas son las que se utilizaron para el modelo. El trabajo a realizar implica testear otras variables en el caso de mermas
features = ['descripcion', 'negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 'tienda', 'motivo', 'ubicacion_motivo', 'merma_monto_p', 'mes', 'anio', 'semestre','dia']
X = data[features]
y = data['merma_unidad_p']

# PASO 4: DIVISIÓN DE DATOS
# 80% entrenamiento, 20% prueba. Este porcentaje es el habitual en la literatura para este tipo de modelos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index] 

mask = X_test.notnull().all(axis=1) & y_test.notnull()
X_test = X_test[mask]
y_test = y_test[mask]

# PASO 5: PREPROCESAMIENTO
# Definir qué variables son categóricas y numéricas
categorical_features = ['descripcion', 'negocio', 'seccion', 'linea', 'categoria','abastecimiento', 'tienda', 'motivo', 'ubicacion_motivo', 'ubicacion_motivo','semestre']
numeric_features = ['mes', 'anio','merma_monto_p','dia']

# Crear preprocesador para manejar ambos tipos de variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# PASO 6: IMPLEMENTACIÓN DE MODELOS
# Modelo 1: Regresión Lineal. Este modelo es el habitual para este tipo de problemas debido a su simplicidad y interpretabilidad.
# En caso de mermas, es posible utilizar este modelo pero pueden explorar otros modelos mas eficientes.
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Modelo 2: Random Forest
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=rf_a_decision, random_state=rf_a_random))
])

# Modelo 3: XGBoost
pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=xgb_a_decision, learning_rate=0.1, max_depth=10, random_state=xgb_a_random))
])

# PASO 7: ENTRENAMIENTO DE MODELOS
# Entrenamos ambos modelos
print("Entrenando Regresión Lineal...")
pipeline_lr.fit(X_train, y_train)

print("Entrenando Random Forest...")
pipeline_rf.fit(X_train, y_train)

print("Entrenando XGBoost...")
pipeline_xgb.fit(X_train, y_train)

print("Modelos entrenados correctamente")

# -------------------------------------------------
# EVALUACIÓN DE LOS MODELOS
# -------------------------------------------------

print("\n=== EVALUACIÓN DE MODELOS PREDICTIVOS ===")

# PASO 8: REALIZAR PREDICCIONES CON LOS MODELOS ENTRENADOS
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_xgb = pipeline_xgb.predict(X_test)

# PASO 9: CALCULAR MÚLTIPLES MÉTRICAS DE EVALUACIÓN
# Error Cuadrático Medio (MSE)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# Raíz del Error Cuadrático Medio (RMSE)
rmse_lr = np.sqrt(mse_lr)
rmse_rf = np.sqrt(mse_rf)
rmse_xgb = np.sqrt(mse_xgb)

# Error Absoluto Medio (MAE)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# Coeficiente de Determinación (R²)
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb)


# NUEVO PASO: GUARDAR RESULTADOS DE PREDICCIÓN EN ARCHIVOS MARKDOWN
# Crear un DataFrame con las predicciones y valores reales
results_df = pd.DataFrame({
    'Valor_Real': y_test,
    'Prediccion_LR': y_pred_lr,
    'Prediccion_RF': y_pred_rf,
    'Prediccion_XGB': y_pred_xgb,
    'Error_LR': y_test - y_pred_lr,
    'Error_RF': y_test - y_pred_rf,
    'Error_XGB': y_test - y_pred_xgb,
    'Error_Porcentual_LR': ((y_test - y_pred_lr) / y_test) * 100,
    'Error_Porcentual_RF': ((y_test - y_pred_rf) / y_test) * 100,
    'Error_Porcentual_XGB': ((y_test - y_pred_xgb) / y_test) * 100

})

# Reiniciar el índice para añadir información de las características
results_df = results_df.reset_index(drop=True)

# Añadir algunas columnas con información de las características para mayor contexto
X_test_reset = X_test.reset_index(drop=True)
for feature in X_test.columns:
    results_df[feature] = X_test_reset[feature]

# Ordenar por valor real para facilitar la comparación
results_df = results_df.sort_values('Valor_Real', ascending=False)

# Guardar resultado para Regresión Lineal
with open('prediccion_lr.md', 'w') as f:
    f.write('# Resultados de Predicción: Regresión Lineal\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_lr:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_lr:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_lr:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    f.write('## Interpretación\n\n')
    f.write(f'El modelo de Regresión Lineal explica aproximadamente el {r2_lr*100:.1f}% de la variabilidad en las ventas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_lr:.2f} unidades.\n\n')
    
    # Mostrar muestra de predicciones (top 10)
    f.write('## Muestra de Predicciones (Top 10)\n\n')
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Linea |\n')
    f.write('|---|------------|------------|-------|---------|-----------|--------|\n')
    
    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i} | {row['Valor_Real']:.2f} | {row['Prediccion_LR']:.2f} | {row['Error_LR']:.2f} | {row['Error_Porcentual_LR']:.1f}% | {row['categoria']} | {row['linea']} |\n")
    
    # Estadísticas de error
    f.write('\n## Distribución del Error\n\n')
    f.write(f'- **Error Mínimo**: {results_df["Error_LR"].min():.2f}\n')
    f.write(f'- **Error Máximo**: {results_df["Error_LR"].max():.2f}\n')
    f.write(f'- **Error Promedio**: {results_df["Error_LR"].mean():.2f}\n')
    f.write(f'- **Desviación Estándar del Error**: {results_df["Error_LR"].std():.2f}\n\n')
    
    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*\n')
print("Archivo de predicción generado: prediccion_lr.md")

# Guardar resultado para Random Forest
with open('prediccion_rf.md', 'w', encoding='utf-8') as f:
    f.write('# Resultados de Predicción: Random Forest\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_rf:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_rf:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_rf:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    f.write('## Interpretación\n\n')
    f.write(f'El modelo de Random Forest explica aproximadamente el {r2_rf*100:.1f}% de la variabilidad en las ventas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_rf:.2f} unidades.\n\n')
    
    f.write('## Muestra de Predicciones (Top 10)\n\n')

    # Encabezados de tabla
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Linea |\n')
    f.write('|---|------------|------------|-------|---------|-----------|--------|\n')

    # Filas de la tabla
    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i+1} | {row['Valor_Real']:.2f} | {row['Prediccion_RF']:.2f} | {row['Error_RF']:.2f} | {row['Error_Porcentual_RF']:.1f}% | {row['categoria']} | {row['linea']} |\n")

    # Estadísticas de error
    f.write('\n## Distribución del Error\n\n')
    f.write(f'- **Error Mínimo**: {results_df["Error_RF"].min():.2f}\n')
    f.write(f'- **Error Máximo**: {results_df["Error_RF"].max():.2f}\n')
    f.write(f'- **Error Promedio**: {results_df["Error_RF"].mean():.2f}\n')
    f.write(f'- **Desviación Estándar del Error**: {results_df["Error_RF"].std():.2f}\n\n')

    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*\n')
print("Archivo de predicción generado: prediccion_rf.md")

    # Crear archivo de predicción
with open('prediccion_xgb.md', 'w', encoding='utf-8') as f:
    f.write('# Resultados de Predicción: XGBoost\n\n')

    # Resumen
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_xgb:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_xgb:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_xgb:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')

    f.write('## Interpretación\n\n')
    f.write(f'El modelo de XGBoost explica aproximadamente el {r2_xgb*100:.1f}% de la variabilidad en las mermas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_xgb:.2f} unidades.\n\n')

    # Muestra top 10
    f.write('## Muestra de Predicciones (Top 10)\n\n')
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Linea |\n')
    f.write('|---|------------|------------|-------|---------|-----------|--------|\n')
    
    # Agrega predicciones de XGBoost al DataFrame
    results_df['Prediccion_XGB'] = y_pred_xgb
    results_df['Error_XGB'] = y_test.reset_index(drop=True) - y_pred_xgb
    results_df['Error_Porcentual_XGB'] = results_df['Error_XGB'] / results_df['Valor_Real'] * 100

    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i+1} | {row['Valor_Real']:.2f} | {row['Prediccion_XGB']:.2f} | {row['Error_XGB']:.2f} | {row['Error_Porcentual_XGB']:.1f}% | {row['categoria']} | {row['linea']} |\n")

    f.write('\n## Distribución del Error\n\n')
    f.write(f'- **Error Mínimo**: {results_df["Error_XGB"].min():.2f}\n')
    f.write(f'- **Error Máximo**: {results_df["Error_XGB"].max():.2f}\n')
    f.write(f'- **Error Promedio**: {results_df["Error_XGB"].mean():.2f}\n')
    f.write(f'- **Desviación Estándar del Error**: {results_df["Error_XGB"].std():.2f}\n\n')
    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real.*\n')
print("Archivo de predicción generado: prediccion_xgb.md")


# PASO 10: PRESENTAR RESULTADOS DE LAS MÉTRICAS EN FORMATO TABULAR
metrics_df = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'Random Forest', 'XGBoost'],
    'MSE': [mse_lr, mse_rf, mse_xgb],
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb],
    'MAE': [mae_lr, mae_rf, mae_xgb],
    'R²': [r2_lr, r2_rf, r2_xgb]
})
print("\nComparación de métricas entre modelos:")
print(metrics_df)

# PASO 11: VISUALIZACIÓN DE PREDICCIONES VS VALORES REALES
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('Random Forest: Predicciones vs Valores Reales')
plt.savefig('predicciones_vs_reales.png')
print("\nGráfico guardado: predicciones_vs_reales.png")

# PASO 12: VISUALIZACIÓN DE RESIDUOS PARA EVALUAR CALIDAD DEL MODELO
residuals = y_test - y_pred_rf
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_rf, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos - Random Forest')
plt.savefig('analisis_residuos.png')
print("Gráfico guardado: analisis_residuos.png")

# PASO 13: DISTRIBUCIÓN DE ERRORES
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribución de Errores - Random Forest')
plt.xlabel('Error')
plt.savefig('distribucion_errores.png')
print("Gráfico guardado: distribucion_errores.png")

# -------------------------------------------------
# DOCUMENTACIÓN DEL PROCESO
# -------------------------------------------------

print("\n=== DOCUMENTACIÓN DEL PROCESO ===")

# PASO 14: DOCUMENTAR LA EXPLORACIÓN INICIAL DE DATOS
print(f"Dimensiones del dataset: {data.shape[0]} filas x {data.shape[1]} columnas")
print(f"Período de tiempo analizado: de {data['anio'].min()} a {data['anio'].max()}")
print(f"Tipos de datos en las columnas principales:")
print(data[features + ['descripcion']].dtypes)

# PASO 15: DOCUMENTAR EL PREPROCESAMIENTO
print("\n--- PREPROCESAMIENTO APLICADO ---")
print(f"Variables numéricas: {numeric_features}")
print(f"Variables categóricas: {categorical_features}")
print("Transformaciones aplicadas:")
print("- Variables numéricas: Estandarización")
print("- Variables categóricas: One-Hot Encoding")

# PASO 16: DOCUMENTAR LA DIVISIÓN DE DATOS
print("\n--- DIVISIÓN DE DATOS ---")
print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/data.shape[0]:.1%} del total)")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/data.shape[0]:.1%} del total)")
print(f"Método de división: Aleatoria con random_state=42")

# PASO 17: DOCUMENTAR LOS MODELOS EVALUADOS
print("\n--- MODELOS IMPLEMENTADOS ---")
print("1. Regresión Lineal:")
print("   - Ventajas: Simple, interpretable")
print("   - Limitaciones: Asume relación lineal entre variables")

print("\n2. Random Forest Regressor:")
print(f"   - Hiperparámetros: n_estimators={rf_a_decision}, random_state={rf_a_random}")
print("   - Ventajas: Maneja relaciones no lineales, menor riesgo de overfitting")
print("   - Limitaciones: Menos interpretable, mayor costo computacional")

print("\n3. XGBoost Regressor:")
print(f"   - Hiperparámetros: n_estimators={xgb_a_decision}, learning_rate=0.1, max_depth=8, random_state={xgb_a_random}")
print("   - Ventajas: Muy eficaz para datos estructurados, maneja relaciones no lineales, buena capacidad para evitar sobreajuste con regularización")
print("   - Limitaciones: Puede ser más lento de entrenar, más complejo de ajustar, requiere más cuidado con hiperparámetros")

# PASO 18: DOCUMENTAR LA VALIDACIÓN DEL MODELO
print("\n--- VALIDACIÓN DEL MODELO ---")
print("Método de validación: Evaluación en conjunto de prueba separado")
print("Métricas utilizadas: MSE, RMSE, MAE, R²")

# PASO 19: VISUALIZAR IMPORTANCIA DE CARACTERÍSTICAS
if hasattr(pipeline_rf['regressor'], 'feature_importances_'):
    print("\n--- IMPORTANCIA DE CARACTERÍSTICAS ---")
    # Obtener nombres de características después de one-hot encoding
    preprocessor = pipeline_rf.named_steps['preprocessor']
    cat_cols = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_cols])
    
    # Obtener importancias
    importances = pipeline_rf['regressor'].feature_importances_
    
    # Crear un DataFrame para visualización
    if len(feature_names) == len(importances):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Mostrar las 10 características más importantes
        print(feature_importance.head(10))
        
        # Visualizar
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Características Más Importantes')
        plt.savefig('importancia_caracteristicas.png')
        print("Gráfico guardado: importancia_caracteristicas.png")
    else:
        print("No se pudo visualizar la importancia de características debido a diferencias en la dimensionalidad")

# PASO 20: CONCLUSIÓN
print("\n=== CONCLUSIÓN ===")

if r2_xgb > r2_rf and r2_xgb > r2_lr:
    print("El mejor modelo según R² es: XGBoost")
    print(f"R² del mejor modelo: {r2_xgb:.4f}")
    print(f"RMSE del mejor modelo: {rmse_xgb:.2f}")
elif r2_rf > r2_lr:
    print("El mejor modelo según R² es: Random Forest")
    print(f"R² del mejor modelo: {r2_rf:.4f}")
    print(f"RMSE del mejor modelo: {rmse_rf:.2f}")
else:
    print("El mejor modelo según R² es: Regresión Lineal")
    print(f"R² del mejor modelo: {r2_lr:.4f}")
    print(f"RMSE del mejor modelo: {rmse_lr:.2f}")

# Explicaciones adicionales para facilitar la interpretación
print("\n--- INTERPRETACIÓN DE RESULTADOS ---")
print(f"• R² (Coeficiente de determinación): Valor entre 0 y 1 que indica qué proporción de la variabilidad")
print(f"  en las mermas/ventas es explicada por el modelo. Un valor de {max(r2_rf, r2_lr,r2_xgb):.4f} significa que")
print(f"  aproximadamente el {max(r2_rf, r2_lr, r2_xgb)*100:.1f}% de la variación puede ser explicada por las variables utilizadas.")

print(f"\n• RMSE (Error cuadrático medio): Representa el error promedio de predicción en las mismas unidades")
if r2_xgb > r2_rf and r2_xgb > r2_lr:
    print(f"  que la variable objetivo. Un RMSE de {rmse_xgb:.2f} significa que, en promedio,")
    print(f"  las predicciones difieren de los valores reales en ±{rmse_xgb:.2f} unidades.")
elif r2_rf > r2_lr:
    print(f"  que la variable objetivo. Un RMSE de {rmse_rf:.2f} significa que, en promedio,")
    print(f"  las predicciones difieren de los valores reales en ±{rmse_rf:.2f} unidades.")
else:
    print(f"  que la variable objetivo. Un RMSE de {rmse_lr:.2f} significa que, en promedio,")
    print(f"  las predicciones difieren de los valores reales en ±{rmse_lr:.2f} unidades.")

print()
if r2_xgb > r2_rf and r2_xgb > r2_lr:
    print("• XGBoost es el mejor modelo porque:")
    print("  - Captura relaciones complejas entre variables")
    print("  - Tiene la mayor capacidad predictiva (R² más alto)")
    print("  - Presenta el menor error de predicción (RMSE más bajo)")
elif r2_rf > r2_lr:
    print("• Random Forest es el mejor modelo porque:")
    print("  - Captura mejor las relaciones no lineales entre las variables")
    print("  - Tiene mayor capacidad predictiva (R² más alto)")
    print("  - Menor error de predicción (RMSE más bajo)")
else:
    print("• Regresión Lineal es el mejor modelo porque:")
    print("  - Ofrece un buen equilibrio entre simplicidad y capacidad predictiva")
    print("  - Es más interpretable que modelos complejos")
    print("  - Presenta un mejor ajuste a los datos en este caso específico")


print("\nEl análisis predictivo ha sido completado exitosamente.") 
