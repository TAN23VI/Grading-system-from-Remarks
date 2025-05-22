import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import spacy
import re
import os
from sklearn.model_selection import GridSearchCV

# === NLP Processing with spaCy 3.8 (used for cleaning in both training and prediction) ===
os.environ['SPACY_WARNING_IGNORE'] = 'W008'
os.environ['TQDM_DISABLE'] = '1'

try:
    nlp = spacy.load("en_core_web_lg", disable=['ner', 'textcat'])
    nlp.max_length = 2000000
    print("SpaCy large model loaded successfully (en_core_web_lg 3.8.0)")
except Exception as e:
    raise Exception(f"Error loading spaCy model 'en_core_web_lg': {str(e)}")

# === Load Transformer Model for Encoding ===
model_path = r"D:\Tanvi\Project\model(LLM)\MiniLM_L6_v2"
try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
except Exception as e:
    raise Exception(f"Error loading transformer model from {model_path}: {str(e)}")

# --- Cleaning Functions ---
def clean_and_correct_text(text):
    """Apply grammar correction cleaning using spaCy"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    tokens = [token.text_with_ws for token in doc]
    cleaned_text = ''.join(tokens)
    cleaned_text = re.sub(r'\s*([.,!?])\s*', r'\1 ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def correct_sentences(sentences):
    return [clean_and_correct_text(s) for s in sentences]

# --- Utility Functions ---
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
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

# === CONFIG ===
main_file = r"X:\Data Transfer  - No Backup\TanviLmb\Ouptut_EndShift_Scoring.xlsx"
other_files = [
    r"X:\Data Transfer  - No Backup\TanviLmb\Ouptut_EndShift_Scoring_Predict.xlsx"
]

scaler = RobustScaler()

mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)

# === Step 1: Train model on main file using grammar-corrected (cleaned) remarks ===
train_df = pd.read_excel(main_file)
main_remark_col = find_remark_column(train_df)
if main_remark_col is None or "Grade" not in train_df.columns:
    raise ValueError("Remark or Grade column missing in main file.")

# Use all rows with both a remark and a grade
train_labeled = train_df.dropna(subset=[main_remark_col, "Grade"])
raw_remarks = train_labeled[main_remark_col].tolist()

# Apply grammar-based cleaning to the main file; this ensures the model learns the corrected language
cleaned_remarks = correct_sentences(raw_remarks)
X_train = encode_sentences(cleaned_remarks, tokenizer, model)
y_train = train_labeled["Grade"].astype(float).values
X_train_scaled = scaler.fit_transform(X_train)
mlp.fit(X_train_scaled, y_train)

print(f"Model trained on {len(train_labeled)} samples using grammar-corrected remark wordings.")

# --- Hyperparameter Tuning ---
# Define the parameter grid for grid search with max_iter set to 1000
param_grid = {
    'hidden_layer_sizes': [(128, 64), (256, 128), (64, 32)],
    'max_iter': [1000]  # Only 1000 iterations for grid search
}

# Create the GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(MLPRegressor(random_state=42),
                           param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error')

# Fit with your training data—using your existing X_train_scaled and y_train
grid_search.fit(X_train_scaled, y_train)

# After search, get the best estimator and parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# === Step 2: Predict and update other files (also apply grammar-based cleaning) ===
for file in other_files:
    df = pd.read_excel(file)
    remark_col = find_remark_column(df)
    if remark_col is None:
        print(f"Skipping {file}: no remark column.")
        continue

    df_remark_only = df.dropna(subset=[remark_col]).copy()
    raw_remarks = df_remark_only[remark_col].tolist()
    # Correct the text using the same grammar-based cleaning
    corrected_remarks = correct_sentences(raw_remarks)
    X = encode_sentences(corrected_remarks, tokenizer, model)
    X_scaled = scaler.transform(X)
    preds = np.rint(mlp.predict(X_scaled)).clip(2, 10).astype(int)

    df.loc[df[remark_col].notna(), "Grade"] = preds

    # Optionally reposition "Grade" next to "END SHIFT REMARK" if that column exists
    if "Grade" in df.columns and "END SHIFT REMARK" in df.columns:
        cols = list(df.columns)
        cols.remove("Grade")
        idx = cols.index("END SHIFT REMARK") + 1
        cols.insert(idx, "Grade")
        df = df[cols]

    # Overwrite the same file with the updated grades
    df.to_excel(file, index=False)
    print(f"Updated file: {file}")



