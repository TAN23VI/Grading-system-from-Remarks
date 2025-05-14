

# Import the necessary packages
import pandas as pd
import transformers
from transformers import pipeline
import scikit-learn
from scikit-learn.model_selection import train_test_split
from scikit-learn.metrics import mean_squared_error
import torch

# Load the Excel sheet
df = pd.read_excel("C:\Users\lmb.bot3\Desktop\Tanvi\Project\Manager_EndShiftRemark_List_13-05-2025 09_59_18.xlsx", sheet_name='Sheet1', usecols=[ 13, 14])  #  PS No and Name are in column 13, Remarks are in column 14

# Extract PS No and Name from the first row
ps_no = df.loc[0, 13]
name = df.loc[0, 13]

# Extract Remarks from the same row as PS No and Name
remarks = df.loc[0, 14]

# Initialize the DistilBERT pipeline for text to vector conversion
embedder = pipeline("text-embedding", model="distilbert-base-uncased")

# Convert the Remarks text into a dense vector
remarks_vector = embedder(remarks)

# Assuming you have an MLP model ready, use it to grade the remark
# Define a simple MLP model if you don't have one
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the MLP model
model = MLP(768, 64, 1)  # Assuming the hidden layer size is 64 and output layer size is 1 for a regression to 0-10 scale
model.to('cuda') if torch.cuda.is_available() else model.to('cpu')

# Load the model weights if you have pre-trained it on a dataset
# model.load_state_dict(torch.load('path_to_your_mlp_weights.pth'))

# Define loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Placeholder for the input vector
input_tensor = torch.tensor(remarks_vector, dtype=torch.float)

# Forward pass
output = model(input_tensor)

# Get the predicted grade
predicted_grade = output.item()

# Print the PS No, Name, Remarks, and Predicted Grade
print(f"PS No: {ps_no}, Name: {name}, Remarks: {remarks}, Predicted Grade: {predicted_grade:.2f}")
```

