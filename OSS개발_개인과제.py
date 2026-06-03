import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # KNN 모델
from sklearn.metrics import accuracy_score

# 1. 데이터셋 로드 및 시각화
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
X, y = data.data, data.target

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("데이터셋 로드 및 분할")
df.head()

# 3. KNN 기본 모델 생성 및 학습
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 4. 예측 및 정확도 출력
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"기본 KNN 모델 테스트 정확도: {accuracy:.4f}")

# 5. 파라미터 변경
experimental_knn = KNeighborsClassifier(n_neighbors=11)
experimental_knn.fit(X_train, y_train)

# 6. 변경된 모델 예측 및 정확도 출력
exp_y_pred = experimental_knn.predict(X_test)
exp_accuracy = accuracy_score(y_test, exp_y_pred)

print(f"실험 KNN 모델 테스트 정확도: {exp_accuracy:.4f}")
