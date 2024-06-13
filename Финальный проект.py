import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest

# Используем бэкэнд Agg
import matplotlib
matplotlib.use('Agg')

# Загрузка данных
file_path = r'C:\Users\dasha\OneDrive\Рабочий стол\Dataset.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Удаление первой строки с заголовками на английском языке
df.columns = df.iloc[0]
df = df.drop(0)

# Переименование колонок для удобства
df.columns = ['Country Name', 'Country Name English', 'AI Index', 'Management Remuneration', 'Management Competence', 'Management Education', 'Digital Skills']

# Преобразование нужных столбцов в числовые типы данных
numeric_columns = ['AI Index', 'Management Remuneration', 'Management Competence', 'Management Education', 'Digital Skills']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Проверка структуры данных
print(df.info())

# Проверка наличия пропущенных значений
print(df.isnull().sum())

# Вывод основных статистик
print(df.describe())

# Построение гистограмм
df[numeric_columns].hist(bins=20, figsize=(15, 10))
plt.savefig('histograms.png')  # Сохранение гистограмм в файл

# Построение коробчатых диаграмм
plt.figure(figsize=(15, 10))
sns.boxplot(data=df[numeric_columns])
plt.xticks(rotation=90)
plt.savefig('boxplots.png')  # Сохранение коробчатых диаграмм в файл

# Построение тепловой карты корреляций
plt.figure(figsize=(15, 10))
corr = df[numeric_columns].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.savefig('heatmap.png')  # Сохранение тепловой карты в файл

# Определение признаков и целевой переменной
X = df[['AI Index', 'Management Remuneration', 'Management Competence', 'Management Education', 'Digital Skills']]
y = df['AI Index']  # Например, если целевая переменная - AI Index

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация моделей
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regressor': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor()
}

# Обучение и оценка моделей
results = {}
feature_importances = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_

# Вывод результатов
for name, metrics in results.items():
    print(f'{name}: MSE = {metrics["MSE"]:.4f}, R2 = {metrics["R2"]:.4f}')

# Визуализация метрик MSE
mse_values = [metrics['MSE'] for metrics in results.values()]
model_names = list(results.keys())

plt.figure(figsize=(10, 5))
plt.barh(model_names, mse_values, color='skyblue')
plt.xlabel('MSE')
plt.title('Mean Squared Error of Different Models')
plt.savefig('mse_comparison.png')  # Сохранение графика сравнения MSE в файл

# Визуализация важности признаков (если доступно)
if feature_importances:
    for name, importances in feature_importances.items():
        plt.figure(figsize=(10, 5))
        plt.barh(X.columns, importances, color='skyblue')
        plt.xlabel('Importance')
        plt.title(f'Feature Importance for {name}')
        plt.savefig(f'{name}_feature_importance.png')

# Применение методов для поиска аномалий
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X)

# Выделение аномалий
mask = yhat == -1
outliers = df[mask]

print(f'Аномалии найдены в следующих данных:\n{outliers}')

# Сохранение аномалий в файл
outliers.to_csv('anomalies.csv', index=False)
