from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from features import build_features


# ========== 读数据 ==========
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
output_dir = base_dir / "outputs"
output_dir.mkdir(exist_ok=True)

train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
print(f"train: {train.shape}, test: {test.shape}")

y = np.log1p(train["SalePrice"])

# ========== 构造特征 ==========
X_train, X_test = build_features(train, test)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ========== 训练最终模型 ==========
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y)

# ========== 预测并还原 ==========
pred_log = model.predict(X_test)
pred = np.expm1(pred_log)  # log1p 的逆运算

# ========== 生成提交文件 ==========
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": pred
})
output_path = output_dir / "submission_random_forest.csv"
submission.to_csv(output_path, index=False)
print(f"已生成 {output_path}")
print(submission.head())
print(f"行数: {len(submission)}")
