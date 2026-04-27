# Developing a Neural Network Regression Model
## Date: 20-04-2025
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1171" height="723" alt="image" src="https://github.com/user-attachments/assets/604fec6d-ccb9-486e-bdec-bd63746efdbb" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: RAMYA R

### Register Number: 212223230169

```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1= nn.Linear(1,8)
        self.fc2= nn.Linear(8,10)
        self.fc3= nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history = {'loss': []}

  def forward(self,x):
    x=self.relu(self.fc1(x))       
    x=self.relu(self.fc2(x)) 
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
# Write your code here
lig = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(lig.parameters(), lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range (epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()

      lig.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

### Dataset Information
<img width="377" height="264" alt="image" src="https://github.com/user-attachments/assets/65c9da6f-bf17-4bd6-9989-a0d0c3e68951" />


<img width="281" height="503" alt="image" src="https://github.com/user-attachments/assets/7fad9c7c-7b07-4cf4-80be-e22f0f0d8b5c" />


### OUTPUT
<img width="493" height="240" alt="image" src="https://github.com/user-attachments/assets/9b9239d6-7c82-44f2-9558-ec0985665cc5" />

### Training Loss Vs Iteration Plot
<img width="795" height="596" alt="image" src="https://github.com/user-attachments/assets/80a2aba9-c6de-4eaf-9d33-cfeffb3fd432" />


### New Sample Data Prediction
<img width="422" height="42" alt="image" src="https://github.com/user-attachments/assets/f766cbe5-a73d-40ee-8a44-5871a1e27d5b" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
