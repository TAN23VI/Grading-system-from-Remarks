import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel
import torch

# Load Excel
file_path = r"C:\Users\lmb.bot3\Desktop\Tanvi\Project\Manager_EndShiftRemark_List_13-05-2025 09_59_18.xlsx"
df = pd.read_excel(file_path)

# Identify the right columns
info_column = df.columns[0]
remarks_column = df.columns[1]

# Extract Info and Remarks
df = df[[info_column, remarks_column]]
df.columns = ['Info', 'Remarks']
df = df.dropna(subset=['Remarks'])

# Dummy grades for training (replace with actual grades if available)
np.random.seed(42)
df['Grade'] = np.random.randint(0, 11, size=len(df))

# Load MiniLM model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Convert remarks to dense vectors
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

print("Generating embeddings...")
df['Embedding'] = df['Remarks'].apply(get_embedding)

# Prepare training data
X = np.stack(df['Embedding'].values)
y = df['Grade'].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train MLP model
print("Training model...")
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Predict grades and round to integers
print("Predicting grades...")
X_all_scaled = scaler.transform(X)
df['Predicted_Grade'] = np.rint(mlp.predict(X_all_scaled)).astype(int)

# Reload original Excel to preserve all formatting and columns
df_output = pd.read_excel(file_path)

# Insert Predicted_Grade column after the Remarks column
remarks_col_idx = df_output.columns.get_loc(remarks_column)
df_output.insert(remarks_col_idx + 1, "Predicted_Grade", df['Predicted_Grade'])

# Save the modified DataFrame to a new Excel file
output_path = file_path.replace(".xlsx", "_with_grades.xlsx")
df_output.to_excel(output_path, index=False)

print(f" Graded file saved with Predicted_Grade beside Remarks at:\n{output_path}")

