#visual embeddings and fine tuning of the transformer is yet to be done
#right now, code reads excel, extracts PS No PS Name and remarks, applies MiniLM, then MLP not linear regression, adds the grade column, presents modified excel
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os


def find_remark_column(df):
    keywords = ["remark", "comment", "note", "feedback"]
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_sentences(sentences, tokenizer, model):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1).cpu().numpy()


# === CONFIG ===
main_file = r"C:\Users\lmb.bot3\Desktop\Tanvi\Project\Manager_EndShiftRemark_List_13-05-2025 09_59_18.xlsx"
other_files = [
    r"C:\Users\lmb.bot3\Desktop\Tanvi\Project\Manager_EndShiftRemark_List_15-05-2025 14_09_48.xlsx",]
    # Add more files if needed


# Load model and tokenizer from local directory
model_path = r"C:\Users\lmb.bot3\Desktop\Tanvi\Project\model\all-MiniLM-L6-v2"
if not os.path.isdir(model_path):
    raise FileNotFoundError(f"Model path not found: {model_path}")

# Load tokenizer and model from local files
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
except Exception as e:
    raise RuntimeError(f"Failed to load model/tokenizer from local path: {model_path}\n{str(e)}")

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

train_labeled = train_df.dropna(subset=[main_remark_col, "Grade"])
X_train = encode_sentences(train_labeled[main_remark_col].tolist(), tokenizer, model)
y_train = train_labeled["Grade"].astype(float).values

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

    df_remark_only = df.dropna(subset=[remark_col]).copy()
    X = encode_sentences(df_remark_only[remark_col].tolist(), tokenizer, model)
    X_scaled = scaler.transform(X)
    preds = np.rint(mlp.predict(X_scaled)).clip(0, 10).astype(int)

    df.loc[df[remark_col].notna(), "Grade"] = preds
    df.to_excel(file, index=False)
    print(f"Updated file with predicted grades: {file}")

