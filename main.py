import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carregar os dados 
df = pd.read_csv('D:/Users/Beatriz/Trabalho/Projeto-Imobiliaria-AD/boston.csv')

# Verificar as primeiras linhas do dataframe
print(df.head())

# Obter informações sobre o dataframe
print(df.info())

# Estatísticas descritivas
print(df.describe())

# Verificar valores faltantes
print(df.isnull().sum())

# Preencher ou remover valores faltantes (exemplo simples)
# Como o conjunto de dados de Boston não tem valores faltantes significativos, podemos ignorar esta parte
# df = df.dropna(axis=1, thresh=int(0.5*len(df)))

# Correlação entre as variáveis
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Scatter plot para algumas variáveis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RM', y='MEDV', data=df)  # Usando 'RM' (número de quartos) e 'MEDV' (preço da casa)
plt.show()

# Selecionar variáveis independentes (features) e dependente (target)
features = ['RM', 'LSTAT', 'PTRATIO']
target = 'MEDV'

X = df[features]
y = df[target]

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# Comparar valores reais e previstos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Comparação entre Valores Reais e Previstos')
plt.show()
