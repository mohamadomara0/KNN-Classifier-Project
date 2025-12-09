import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# تحميل البيانات
df = pd.read_csv('simple_iris_dataset.csv')

# فصل الخصائص عن الهدف
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# توحيد القيم (Normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# بناء النموذج
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# التنبؤ
y_pred = knn.predict(X_test_scaled)

# التقييم
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))