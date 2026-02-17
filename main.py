import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)
import time
from IPython.display import display

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 14

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

print("Библиотеки импортированы, настройки применены.")

# ==================================================
# 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
# ==================================================

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("=" * 80)
print("РАЗМЕР ДАТАСЕТА")
print("=" * 80)
print(f"Количество строк: {df.shape[0]:,}")
print(f"Количество столбцов: {df.shape[1]}")

print("\n" + "=" * 80)
print("ПЕРВЫЕ 5 СТРОК")
print("=" * 80)
display(df.head())

print("\n" + "=" * 80)
print("ИНФОРМАЦИЯ О СТОЛБЦАХ")
print("=" * 80)
df.info()

print("\n" + "=" * 80)
print("ТИПЫ ДАННЫХ")
print("=" * 80)
print(df.dtypes.value_counts())

print("\n" + "=" * 80)
print("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
print("=" * 80)
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("Пропущенных значений нет.")
else:
    display(missing)

print("\n" + "=" * 80)
print("ДУБЛИКАТЫ")
print("=" * 80)
print(f"Количество полных дубликатов: {df.duplicated().sum()}")

# ==================================================
# 2. АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ==================================================

print("\n" + "=" * 80)
print("ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (Churn)")
print("=" * 80)

churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
print("Абсолютные значения:")
print(churn_counts)
print("\nОтносительные значения:")
print(churn_pct.round(1))

plt.figure(figsize=(8, 6))
colors = ['#2E8B57', '#DC143C']
ax = churn_pct.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Распределение целевой переменной Churn', fontsize=16, fontweight='bold')
plt.xlabel('Churn')
plt.ylabel('Процент клиентов')
plt.xticks(rotation=0)
for i, v in enumerate(churn_pct.values):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ==================================================
# 3. КЛАССИФИКАЦИЯ ПРИЗНАКОВ
# ==================================================

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'customerID']
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\n" + "=" * 80)
print(f"КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ({len(categorical_cols)})")
print("=" * 80)
print(categorical_cols)

print("\n" + "=" * 80)
print(f"ЧИСЛОВЫЕ ПРИЗНАКИ ({len(numerical_cols)})")
print("=" * 80)
print(numerical_cols)

print("\n" + "=" * 80)
print("УНИКАЛЬНЫЕ ЗНАЧЕНИЯ В КАТЕГОРИАЛЬНЫХ ПРИЗНАКАХ")
print("=" * 80)
for col in categorical_cols:
    uniq = df[col].unique()
    print(f"{col}: {len(uniq)} уникальных значений -> {list(uniq)}")

# ==================================================
# 4. ПРОВЕРКА ГИПОТЕЗ (EDA)
# ==================================================

# Гипотеза 1: Тип контракта
print("\n" + "=" * 80)
print("ГИПОТЕЗА 1: ВЛИЯНИЕ ТИПА КОНТРАКТА")
print("=" * 80)
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
print("Процент оттока по типам контракта:")
display(contract_churn.style.format("{:.1f}%"))

plt.figure(figsize=(10, 6))
contract_churn.plot(kind='bar', stacked=True, color=['#2E8B57', '#DC143C'], edgecolor='black')
plt.title('Отток по типам контракта')
plt.xlabel('Тип контракта')
plt.ylabel('Процент')
plt.legend(title='Churn')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Гипотеза 2: Способ оплаты
print("\n" + "=" * 80)
print("ГИПОТЕЗА 2: ВЛИЯНИЕ СПОСОБА ОПЛАТЫ")
print("=" * 80)
payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
print("Процент оттока по способам оплаты:")
display(payment_churn.style.format("{:.1f}%"))

plt.figure(figsize=(12, 6))
payment_churn.sort_values('Yes', ascending=True).plot(
    kind='barh', stacked=True, color=['#2E8B57', '#DC143C'], edgecolor='black'
)
plt.title('Отток по способам оплаты')
plt.xlabel('Процент')
plt.ylabel('Способ оплаты')
plt.legend(title='Churn')
plt.tight_layout()
plt.show()

# Гипотеза 3: Ежемесячные платежи
print("\n" + "=" * 80)
print("ГИПОТЕЗА 3: ВЛИЯНИЕ ЕЖЕМЕСЯЧНЫХ ПЛАТЕЖЕЙ")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette=colors, ax=axes[0])
axes[0].set_title('Ежемесячные платежи по оттоку')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Monthly Charges ($)')

sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, bins=30, palette=colors, ax=axes[1])
axes[1].set_title('Распределение ежемесячных платежей')
axes[1].set_xlabel('Monthly Charges ($)')
axes[1].set_ylabel('Количество клиентов')

plt.tight_layout()
plt.show()

stats = df.groupby('Churn')['MonthlyCharges'].agg(['mean', 'median', 'std'])
print("Статистика ежемесячных платежей:")
display(stats.round(2))

# ==================================================
# 5. ПРЕДОБРАБОТКА ДАННЫХ
# ==================================================

print("\n" + "=" * 80)
print("ПРЕДОБРАБОТКА: ОБРАБОТКА TotalCharges")
print("=" * 80)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"Пропусков после преобразования: {df['TotalCharges'].isnull().sum()}")

df.loc[df['tenure'] == 0, 'TotalCharges'] = df.loc[df['tenure'] == 0, 'MonthlyCharges']
df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
print(f"Пропусков после заполнения: {df['TotalCharges'].isnull().sum()}")

print("\n" + "=" * 80)
print("ПРЕДОБРАБОТКА: СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ")
print("=" * 80)

internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
df['InternetServicesCount'] = 0
for service in internet_services:
    df['InternetServicesCount'] += (df[service] == 'Yes').astype(int)
print("InternetServicesCount добавлен.")

df['HasFamily'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
print("HasFamily добавлен.")

df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
print("AvgMonthlySpend добавлен.")

premium_services = ['OnlineSecurity', 'TechSupport', 'DeviceProtection']
df['PremiumServices'] = 0
for service in premium_services:
    df['PremiumServices'] += (df[service] == 'Yes').astype(int)
print("PremiumServices добавлен.")

print(f"\nПропусков после создания признаков: {df.isnull().sum().sum()}")

# ==================================================
# 6. КОДИРОВАНИЕ ПРИЗНАКОВ
# ==================================================

print("\n" + "=" * 80)
print("КОДИРОВАНИЕ ПРИЗНАКОВ")
print("=" * 80)

df_encoded = pd.DataFrame()
df_encoded['Churn'] = (df['Churn'] == 'Yes').astype(int)

num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
                'InternetServicesCount', 'HasFamily', 'AvgMonthlySpend', 'PremiumServices']
for feat in num_features:
    df_encoded[feat] = df[feat]

df_encoded['gender'] = (df['gender'] == 'Male').astype(int)
df_encoded['Partner'] = (df['Partner'] == 'Yes').astype(int)
df_encoded['Dependents'] = (df['Dependents'] == 'Yes').astype(int)
df_encoded['PhoneService'] = (df['PhoneService'] == 'Yes').astype(int)
df_encoded['PaperlessBilling'] = (df['PaperlessBilling'] == 'Yes').astype(int)

contract_dummies = pd.get_dummies(df['Contract'], prefix='Contract', drop_first=True)
internet_dummies = pd.get_dummies(df['InternetService'], prefix='InternetService', drop_first=True)
payment_dummies = pd.get_dummies(df['PaymentMethod'], prefix='PaymentMethod', drop_first=True)

df_encoded = pd.concat([df_encoded, contract_dummies, internet_dummies, payment_dummies], axis=1)

print(f"Размерность после кодирования: {df_encoded.shape}")
print(f"Пропусков: {df_encoded.isnull().sum().sum()}")

# ==================================================
# 7. МАСШТАБИРОВАНИЕ
# ==================================================

print("\n" + "=" * 80)
print("МАСШТАБИРОВАНИЕ ЧИСЛОВЫХ ПРИЗНАКОВ")
print("=" * 80)

y = df_encoded['Churn']
X = df_encoded.drop('Churn', axis=1)

numerical_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend',
                      'InternetServicesCount', 'SeniorCitizen', 'PremiumServices']

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_to_scale] = scaler.fit_transform(X[numerical_to_scale])

print("Числовые признаки отмасштабированы.")
print(f"Пропусков после масштабирования: {X_scaled.isnull().sum().sum()}")
print(f"Финальный размер X: {X_scaled.shape}")

# ==================================================
# 8. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# ==================================================

print("\n" + "=" * 80)
print("РАЗДЕЛЕНИЕ ДАННЫХ")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]:,} строк")
print(f"Тестовая выборка: {X_test.shape[0]:,} строк")
print(f"Распределение классов сохранено.")

# ==================================================
# 9. ОБУЧЕНИЕ МОДЕЛЕЙ
# ==================================================

print("\n" + "=" * 80)
print("МОДЕЛЬ 1: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
print("=" * 80)

lr = LogisticRegression(random_state=42, max_iter=1000)
start = time.time()
lr.fit(X_train, y_train)
train_time = time.time() - start
print(f"Время обучения: {train_time:.3f} сек")

y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_proba_lr)

print(f"Accuracy:  {acc_lr:.4f}")
print(f"Precision: {prec_lr:.4f}")
print(f"Recall:    {rec_lr:.4f}")
print(f"F1-score:  {f1_lr:.4f}")
print(f"ROC-AUC:   {auc_lr:.4f}")

print("\n" + "=" * 80)
print("МОДЕЛЬ 2: СЛУЧАЙНЫЙ ЛЕС")
print("=" * 80)

rf = RandomForestClassifier(random_state=42, n_estimators=100)
start = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start
print(f"Время обучения: {train_time:.3f} сек")

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

print(f"Accuracy:  {acc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall:    {rec_rf:.4f}")
print(f"F1-score:  {f1_rf:.4f}")
print(f"ROC-AUC:   {auc_rf:.4f}")

print("\n" + "=" * 80)
print("МОДЕЛЬ 3: ГРАДИЕНТНЫЙ БУСТИНГ")
print("=" * 80)

gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
start = time.time()
gb.fit(X_train, y_train)
train_time = time.time() - start
print(f"Время обучения: {train_time:.3f} сек")

y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:, 1]

acc_gb = accuracy_score(y_test, y_pred_gb)
prec_gb = precision_score(y_test, y_pred_gb)
rec_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
auc_gb = roc_auc_score(y_test, y_proba_gb)

print(f"Accuracy:  {acc_gb:.4f}")
print(f"Precision: {prec_gb:.4f}")
print(f"Recall:    {rec_gb:.4f}")
print(f"F1-score:  {f1_gb:.4f}")
print(f"ROC-AUC:   {auc_gb:.4f}")

# ==================================================
# 10. СРАВНЕНИЕ МОДЕЛЕЙ
# ==================================================

print("\n" + "=" * 80)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 80)

comparison = pd.DataFrame({
    'Logistic Regression': [acc_lr, prec_lr, rec_lr, f1_lr, auc_lr],
    'Random Forest': [acc_rf, prec_rf, rec_rf, f1_rf, auc_rf],
    'Gradient Boosting': [acc_gb, prec_gb, rec_gb, f1_gb, auc_gb]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'])

print("Сводная таблица метрик:")
display(comparison.style.format("{:.4f}").highlight_max(axis=1, color='lightgreen'))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
x = np.arange(len(metrics))
width = 0.25
ax = axes[0,0]
ax.bar(x - width, [acc_lr, prec_lr, rec_lr, f1_lr], width, label='LogReg', color='#3498db')
ax.bar(x, [acc_rf, prec_rf, rec_rf, f1_rf], width, label='RF', color='#2ecc71')
ax.bar(x + width, [acc_gb, prec_gb, rec_gb, f1_gb], width, label='GB', color='#e74c3c')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Значение')
ax.set_title('Сравнение метрик')
ax.legend()

ax = axes[0,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
ax.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC={auc_lr:.3f})', color='#3498db')
ax.plot(fpr_rf, tpr_rf, label=f'RF (AUC={auc_rf:.3f})', color='#2ecc71')
ax.plot(fpr_gb, tpr_gb, label=f'GB (AUC={auc_gb:.3f})', color='#e74c3c')
ax.plot([0,1],[0,1],'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC-кривые')
ax.legend(loc='lower right')

ax = axes[1,0]
fn_vals = [
    confusion_matrix(y_test, y_pred_lr).ravel()[2],
    confusion_matrix(y_test, y_pred_rf).ravel()[2],
    confusion_matrix(y_test, y_pred_gb).ravel()[2]
]
ax.bar(['LogReg', 'RF', 'GB'], fn_vals, color=['#3498db','#2ecc71','#e74c3c'])
ax.set_ylabel('Количество False Negative')
ax.set_title('Пропущенные уходы (FN)')

ax = axes[1,1]
best_model = gb
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
ax.barh(range(10), importances[indices][::-1], color='#3498db')
ax.set_yticks(range(10))
ax.set_yticklabels(X.columns[indices][::-1])
ax.set_xlabel('Важность')
ax.set_title('Топ-10 важных признаков (Gradient Boosting)')

plt.tight_layout()
plt.show()

# ==================================================
# 11. ИНТЕРПРЕТАЦИЯ И ВЫВОДЫ
# ==================================================

print("\n" + "=" * 80)
print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 80)

best_f1 = max(f1_lr, f1_rf, f1_gb)
if best_f1 == f1_gb:
    best_name = 'Gradient Boosting'
    best_model = gb
    best_rec = rec_gb
    cm = confusion_matrix(y_test, y_pred_gb)
elif best_f1 == f1_rf:
    best_name = 'Random Forest'
    best_model = rf
    best_rec = rec_rf
    cm = confusion_matrix(y_test, y_pred_rf)
else:
    best_name = 'Logistic Regression'
    best_model = lr
    best_rec = rec_lr
    cm = confusion_matrix(y_test, y_pred_lr)

print(f"Лучшая модель по F1-score: {best_name} (F1 = {best_f1:.4f})")
print(f"Recall (доля найденных уходящих): {best_rec:.4f} ({best_rec*100:.1f}%)")

print("\nМатрица ошибок:")
cm_df = pd.DataFrame(cm,
                     index=['Факт: остался', 'Факт: ушёл'],
                     columns=['Прогноз: остался', 'Прогноз: ушёл'])
display(cm_df)

tn, fp, fn, tp = cm.ravel()
print(f"Верно предсказано осталось: {tn}")
print(f"Ложная тревога (ошибочно предсказан уход): {fp}")
print(f"Пропущено уходов: {fn}")
print(f"Верно предсказано уходов: {tp}")

print("\nКлючевые факторы оттока (топ-5):")
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
    top5 = feat_imp.sort_values(ascending=False).head(5)
    for feat, imp in top5.items():
        print(f"  - {feat}: {imp:.4f} ({imp*100:.1f}%)")
else:
    coef = pd.Series(best_model.coef_[0], index=X.columns)
    top5_pos = coef.sort_values(ascending=False).head(5)
    top5_neg = coef.sort_values().head(5)
    print("Положительное влияние (увеличивают отток):")
    for feat, val in top5_pos.items():
        print(f"  - {feat}: {val:.4f}")
    print("Отрицательное влияние (снижают отток):")
    for feat, val in top5_neg.items():
        print(f"  - {feat}: {val:.4f}")

print("\n" + "=" * 80)
print("ПРОЕКТ ВЫПОЛНЕН. ДАННЫЕ ПОДГОТОВЛЕНЫ, МОДЕЛИ ОБУЧЕНЫ.")
print("=" * 80)