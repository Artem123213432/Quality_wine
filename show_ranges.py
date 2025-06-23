import pandas as pd

def main():
    # Загружаем данные
    wine_data = pd.read_csv("data/winequality-red.csv", sep=';')
    
    # Выводим статистические данные по каждому признаку
    print("Диапазоны значений для каждого признака вина:")
    print("=============================================")
    
    # Используем describe для получения статистики
    stats = wine_data.describe()
    
    for feature in wine_data.columns[:-1]:  # Исключаем целевую переменную quality
        min_val = stats.loc['min', feature]
        max_val = stats.loc['max', feature]
        mean_val = stats.loc['mean', feature]
        median_val = stats.loc['50%', feature]
        
        print(f"\n{feature}:")
        print(f"  Минимум:  {min_val:.3f}")
        print(f"  Максимум: {max_val:.3f}")
        print(f"  Среднее:  {mean_val:.3f}")
        print(f"  Медиана:  {median_val:.3f}")
    
    # Показываем распределение оценок качества
    quality_counts = wine_data['quality'].value_counts().sort_index()
    
    print("\n\nРаспределение оценок качества в наборе данных:")
    print("=============================================")
    for quality, count in quality_counts.items():
        print(f"Качество {quality}: {count} образцов")
    
    # Выводим сводную информацию о размере набора данных
    print(f"\nВсего образцов в наборе данных: {len(wine_data)}")

if __name__ == "__main__":
    main() 