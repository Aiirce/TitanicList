import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# --- RUTA DEL ARCHIVO ---
file_path = r"C:\Users\rodol\Downloads\titanic.csv"
df = pd.read_csv(file_path)

# HEAD
print(df.head())

# INFO
print("\nInfo del DataFrame")
df.info()

# DESCRIBE
print("\nDF Describe")
print(df.describe())

# COLUMNAS
print("\nColumnas del DataFrame")
print(df.columns.tolist())

# Conteo de sobrevivientes
print("\nConteo de sobrevivientes (0 = No, 1 = Sí)")
print(df['Survived'].value_counts())

# Conteo por clase
print("\nConteo de pasajeros por clase")
print(df['Pclass'].value_counts().sort_index())

# Conteo por sexo
print("\nConteo de pasajeros por sexo")
print(df['Sex'].value_counts())

# Valores nulos
print("\nValores nulos por columna")
print(df.isnull().sum())

# Gráficas
sns.countplot(data=df, x='Survived')
plt.title("Supervivencia en el Titanic (0=No, 1=Sí)")
plt.show()

sns.countplot(data=df, x='Pclass')
plt.title("Distribución por clase")
plt.show()

sns.countplot(data=df, x='Sex')
plt.title("Sexo de los pasajeros")
plt.show()

# -------- Funciones --------
def tasa_supervivencia_por_clase(df):
    return df.groupby('Pclass')['Survived'].mean()

def tasa_supervivencia_por_sexo(df):
    return df.groupby('Sex')['Survived'].mean()

def pasajeros_menores(df, edad=18):
    return df[df['Age'] < edad][['Name', 'Age', 'Sex', 'Pclass', 'Survived']]

def top_tarifas(df, n=5):
    return df[['Name', 'Fare', 'Pclass', 'Survived']].sort_values(by='Fare', ascending=False).head(n)

def pasajeros_por_puerto(df, puerto):
    return df[df['Embarked'] == puerto][['Name', 'Sex', 'Pclass', 'Survived']]

# Estadísticos de Age
media = df['Age'].mean()
moda = df['Age'].mode()[0]
mediana = df['Age'].median()
desviacion = df['Age'].std()
print(f"\nMedia edad: {media}")
print(f"Moda edad: {moda}")
print(f"Mediana edad: {mediana}")
print(f"Desviación estándar edad: {desviacion}")

print("\nTasa de supervivencia por clase:")
print(tasa_supervivencia_por_clase(df))

print("\nTasa de supervivencia por sexo:")
print(tasa_supervivencia_por_sexo(df))

print("\nPasajeros menores de 18 (primeras 5 filas):")
print(pasajeros_menores(df, 18).head())

print("\nPasajeros con tarifas más altas:")
print(top_tarifas(df, 5))

print("\nPasajeros que embarcaron en Southampton:")
print(pasajeros_por_puerto(df, 'S').head())
