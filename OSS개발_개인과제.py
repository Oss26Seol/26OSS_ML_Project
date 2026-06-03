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
