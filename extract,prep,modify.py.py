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
        if any(k in col.lower() for k in keywords):
            return col
    return None
 
# === CONFIG ===
main_file = r"C:\Users\20004806\Documents\Python Scripts\Smart End Shift\Tanvi\Manager_EndShiftRemark_List_13-05-2025 09_59_18.xlsx"
other_files = [
    r"C:\Users\20004806\Documents\Python Scripts\Smart End Shift\Tanvi\Manager_EndShiftRemark_List_15-05-2025 14_09_48.xlsx"
    # Add more files if needed
]
 
# Load SentenceTransformer model (online mode)
model = SentenceTransformer("all-MiniLM-L6-v2")
 
# Initialize scaler and ML model
scaler = MinMaxScaler()
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
 
# === Step 1: Train model on main file ===
train_df = pd.read_excel(main_file)
main_remark_col = find_remark_column(train_df)
if main_remark_col is None:
    raise ValueError(f"No remark-like column found in {main_file}")
 
if "Grade" not in train_df.columns:
    raise ValueError(f"'Grade' column not found in training file: {main_file}")
 
# Drop rows with missing remarks or grades
train_labeled = train_df.dropna(subset=[main_remark_col, "Grade"])
X_train = np.stack(train_labeled[main_remark_col].apply(model.encode))
y_train = train_labeled["Grade"].astype(float).values
 
# Fit the scaler and model
X_train_scaled = scaler.fit_transform(X_train)
mlp.fit(X_train_scaled, y_train)
 
print(f"Model trained on {len(train_labeled)} samples from: {main_file}")
 
# === Step 2: Predict and update other files ===
for file in other_files:
    df = pd.read_excel(file)
    remark_col = find_remark_column(df)
 
    if remark_col is None:
        print(f"Skipping {file}: no remark-like column found.")
        continue
 
    # Keep only rows where remark is not blank
    df_remark_only = df.dropna(subset=[remark_col]).copy()
 
    # Encode and predict
    X = np.stack(df_remark_only[remark_col].apply(model.encode))
    X_scaled = scaler.transform(X)
    preds = np.rint(mlp.predict(X_scaled)).clip(0, 10).astype(int)  # Ensure scale is 0–10
 
    # Add predictions back to full DataFrame
    df.loc[df[remark_col].notna(), "Grade"] = preds
 
    # Save updated file
    df.to_excel(file, index=False)
    print(f"Updated file with predicted grades: {file}")
