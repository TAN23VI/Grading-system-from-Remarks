#visual embeddings and fine tuning of the transformer is yet to be done
#right now, code reads excel, extracts PS No PS Name and remarks, applies MiniLM, then MLP not linear regression, adds the grade column, presents modified excel

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

def find_remark_column(df):
    keywords = ["remark", "comment", "note", "feedback"]
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in keywords):
            return col
    return None

# === CONFIG ===
main_file = "Manager_EndShiftRemark_List_13-05-2025 09_59_18.xlsx"
other_files = [
    "complex_worker_remarks.xlsx",
    "Manager_EndShiftRemark_List_Demo.xlsx",
    # Add other files here
]

# Load model and scaler (offline mode)
model_path = r"C:\Users\lmb.bot3\Desktop\Tanvi\Project\model\all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)
scaler = MinMaxScaler()
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)

# === Step 1: Load main file and find remark column ===
train_df = pd.read_excel(main_file)
main_remark_col = find_remark_column(train_df)
if main_remark_col is None:
    raise ValueError(f"No remark-like column found in {main_file}")
print(f"Detected remark column in main file: '{main_remark_col}'")

# Prepare training data
if train_df["Grade"].isna().sum() == 0:
    # All grades present, train on all
    print(f"All grades present in {main_file}, training model on full data.")
    X_train = np.stack(train_df[main_remark_col].apply(model.encode))
    y_train = train_df["Grade"].astype(float).values
    mlp.fit(scaler.fit_transform(X_train), y_train)
else:
    train_labeled = train_df.dropna(subset=[main_remark_col, "Grade"])
    X_train = np.stack(train_labeled[main_remark_col].apply(model.encode))
    y_train = train_labeled["Grade"].astype(float).values
    mlp.fit(scaler.fit_transform(X_train), y_train)
    
    # Predict missing grades in main file
    to_predict = train_df[train_df["Grade"].isna()]
    if not to_predict.empty:
        X_missing = np.stack(to_predict[main_remark_col].apply(model.encode))
        X_missing_scaled = scaler.transform(X_missing)
        preds = np.rint(mlp.predict(X_missing_scaled)).astype(int)
        train_df.loc[train_df["Grade"].isna(), "Grade"] = preds
        print(f"Filled {len(preds)} missing grades in {main_file}")
    
    # Save updated main file
    train_df.to_excel(main_file, index=False)

# === Step 2: Predict for other files ===
for file in other_files:
    df = pd.read_excel(file)
    remark_col = find_remark_column(df)
    if remark_col is None:
        print(f"Skipping {file}: no remark-like column found.")
        continue

    df = df.dropna(subset=[remark_col])
    X = np.stack(df[remark_col].apply(model.encode))
    X_scaled = scaler.transform(X)

    preds = np.rint(mlp.predict(X_scaled)).astype(int)
    df["Grade"] = preds

    df.to_excel(file, index=False)
    print(f"Processed and updated: {file}")


