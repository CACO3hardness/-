import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def build_features(train, test):
    """把 train 和 test 一起做特征工程，保证编码一致"""
    train_df = train.drop(columns=["SalePrice"]).copy()
    test_df = test.copy()

    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    all_df = all_df.drop(columns=["Id"])

    num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = all_df.select_dtypes(include=["object"]).columns.tolist()

    all_df[num_cols] = all_df[num_cols].fillna(all_df[num_cols].median())
    all_df[cat_cols] = all_df[cat_cols].fillna("None")

    all_df["TotalSF"] = all_df["TotalBsmtSF"] + all_df["1stFlrSF"] + all_df["2ndFlrSF"]
    all_df["TotalBath"] = (
        all_df["FullBath"] + 0.5 * all_df["HalfBath"]
        + all_df["BsmtFullBath"] + 0.5 * all_df["BsmtHalfBath"]
    )
    all_df["HouseAge"] = all_df["YrSold"] - all_df["YearBuilt"]
    all_df["RemodAge"] = all_df["YrSold"] - all_df["YearRemodAdd"]
    all_df["TotalPorchSF"] = (
        all_df["OpenPorchSF"] + all_df["EnclosedPorch"]
        + all_df["3SsnPorch"] + all_df["ScreenPorch"]
    )
    all_df["HasPool"] = (all_df["PoolArea"] > 0).astype(int)
    all_df["Has2ndFloor"] = (all_df["2ndFlrSF"] > 0).astype(int)
    all_df["HasGarage"] = (all_df["GarageArea"] > 0).astype(int)
    all_df["HasBsmt"] = (all_df["TotalBsmtSF"] > 0).astype(int)
    all_df["HasFireplace"] = (all_df["Fireplaces"] > 0).astype(int)

    all_df = pd.get_dummies(all_df, columns=cat_cols, drop_first=True)

    n_train = len(train_df)
    X_train = all_df.iloc[:n_train].reset_index(drop=True)
    X_test = all_df.iloc[n_train:].reset_index(drop=True)
    return X_train, X_test


# ========== 读数据 ==========
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print(f"train: {train.shape}, test: {test.shape}")

y = np.log1p(train["SalePrice"])

# ========== 构造特征 ==========
X_train, X_test = build_features(train, test)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ========== 训练最终模型（exp7 最优配置）==========
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=7,
    random_state=42
)
model.fit(X_train, y)

# ========== 预测并还原 ==========
pred_log = model.predict(X_test)
pred = np.expm1(pred_log)

# ========== 生成提交文件 ==========
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": pred
})
submission.to_csv("submission.csv", index=False)
print("已生成 submission.csv")
print(submission.head())
print(f"行数: {len(submission)}")