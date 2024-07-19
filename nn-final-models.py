#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#       CNN Model , From Scratch  


# In[1]:


#################################       CNN Model , From Scratch     #################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import numpy as np
import cv2
import os

# Preprocessing and loading data
class_names = ["Apples", "Banana", "Grapes", "Mango", "Strawberry"]
path = "/kaggle/input/mosaleh/dataset NN'23/dataset/train"

images_train = []
labels = []

image_size = (224, 224)

Apples = 0
Banana = 0
Grapes = 0
Mango = 0
Strawberry = 0

for subdir in os.listdir(path):
    subdir_path = os.path.join(path, subdir)
    if os.path.isdir(subdir_path):
        for img in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img)

            try:
                image = cv2.imread(img_path)
                if image is None:
                    raise Exception(f"Error reading image: {img_path}")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size)

                images_train.append(image)

                fruit_name = class_names[int(subdir) - 1]
                if fruit_name == "Apples":
                    labels.append(1)
                    Apples += 1
                elif fruit_name == "Banana":
                    labels.append(2)
                    Banana += 1
                elif fruit_name == "Grapes":
                    labels.append(3)
                    Grapes += 1
                elif fruit_name == "Mango":
                    labels.append(4)
                    Mango += 1
                elif fruit_name == "Strawberry":
                    labels.append(5)
                    Strawberry += 1

            except Exception as e:
                print(e)

labels = np.array(labels) - 1
images_train = np.array(images_train, dtype='float32') / 255.0
print(images_train.shape)

x_train, x_test, y_train, y_test = train_test_split(images_train, labels, random_state=5, shuffle=True, test_size=0.3)

class CustomNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.flatten(start_dim=1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Augmentation transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

# Apply augmentation transforms to training data
augmented_data = [(transform(Image.fromarray((image * 255).astype(np.uint8))), label) for image, label in zip(x_train, y_train)]
x_train_augmented, y_train_augmented = zip(*augmented_data)

# Convert to tensors
x_train_tensor = torch.stack(x_train_augmented)
y_train_tensor = torch.Tensor(y_train_augmented).long()
batch_size = 128
# Create DataLoader for augmented training data
augmented_train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model
model = CustomNet()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(20):
    model.train()
    correct_train = 0
    total_train = 0
    for images, labels in augmented_train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    train_accuracy = correct_train / total_train
    print(f'Training Accuracy for Epoch {epoch + 1}: {train_accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), "custom_model_with_augmentation.pth")

# Convert the test data to tensors
x_test_tensor = torch.Tensor(x_test).permute(0, 3, 1, 2)  # Adjust the dimensions
y_test_tensor = torch.Tensor(y_test).long()

# Create DataLoader for test data
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle for evaluation

# Evaluate the model
model.eval()
predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

accuracy = (np.array(predictions) == y_test).mean()
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# Create confusion matrix (optional)
cm = confusion_matrix(y_test, predictions)
cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)
cm_display.plot()



# -------------------------------------------------------------------------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# -------------------------------------------------------------------------------------------------------------------

#                <***************************>      Test Model     <**********************************>  csv result 


import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms

# Load the PyTorch model
model = CustomNet()
model.load_state_dict(torch.load("custom_model_with_augmentation.pth"))  # Load the model with data augmentation
model.to(device)  # Move the model to the same device as the input data
model.eval()

# Testing
class_names = ["Apples", "Banana", "Grapes", "Mango", "Strawberry"]

image_name = []
images_test = []

# Assuming '/path/to/test/' is the path to the directory containing test images
test_path = "/kaggle/input/mosaleh/dataset NN'23/dataset/test"

for img in os.listdir(test_path):
    img_path = os.path.join(test_path, img)

    try:
        image = cv2.imread(img_path)
        if image is None:
            raise Exception(f"Error reading image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  # Assuming input shape of your custom model is (224, 224, 3)

        # Apply the same data preprocessing steps as during training
        image = image.astype(np.float32) / 255.0

        images_test.append(image)
        image_name.append(img)

    except Exception as e:
        print(e)

images_test = np.array(images_test, dtype='float32')

# Convert images_test to a PyTorch tensor
images_tensor = torch.from_numpy(images_test.transpose(0, 3, 1, 2))
images_tensor = images_tensor.to(device)  # Assuming you defined the device variable

# Create a dummy labels array for test_loader (you can replace this with your actual labels)
dummy_labels = np.zeros(len(images_test))

# Create a DataLoader for test data
test_dataset = TensorDataset(images_tensor, torch.from_numpy(dummy_labels))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Make predictions
predictions = []
with torch.no_grad():
    model.eval()
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# Convert predictions to numpy array
predictions = np.array(predictions)

# Extract image numbers without extension
image_numbers = [int(img.split('.')[0]) for img in image_name]

# Add 1 to the predicted labels if you want to align with the original indexing (1 to 5)
predict_label = predictions + 1

# Save predictions to a CSV file without an index
dict_target = {"image_id": image_numbers, "label": predict_label}
df = pd.DataFrame(dict_target)

# Save the CSV file
csv_filename = "Cnn_with_augmentation.csv"
df.to_csv(csv_filename, index=False)

# Provide a downloadable link
print(f"CSV file saved as {csv_filename}")
print(f"Download link: [Download CSV](sandbox:/path/to/{csv_filename})")

import os

current_directory = os.getcwd()
print("Current Directory:", current_directory)

df.to_csv("/kaggle/working/Cnn_with_augmentation.csv", index=False)

# Load the CSV file
df_predictions = pd.read_csv("Cnn_with_augmentation.csv")

# Display the contents of the DataFrame
display(df_predictions)

# Provide a download link
from IPython.display import FileLink

# Provide a download link for the CSV file
FileLink(r'Cnn_with_augmentation.csv')


# In[ ]:





# In[ ]:


# Rez_Net_50_Pretrain & Scratch :>>>>>>>


# In[ ]:


#         ################################>>>>>>>>>>        Rez_Net_50_Pretrain & Scratch        <<<<<<<<<<<<<<###########################


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

# Preprocessing and loading data
class_names = ["Apples", "Banana", "Grapes", "Mango", "Strawberry"]
path = "/kaggle/input/mosaleh/dataset NN'23/dataset/train"

images_train = []
labels = []

image_size = (224, 224)

Apples = 0
Banana = 0
Grapes = 0
Mango = 0
Strawberry = 0

for subdir in os.listdir(path):
    subdir_path = os.path.join(path, subdir)
    if os.path.isdir(subdir_path):
        for img in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img)

            try:
                image = cv2.imread(img_path)
                if image is None:
                    raise Exception(f"Error reading image: {img_path}")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size)

                images_train.append(image)

                fruit_name = class_names[int(subdir) - 1]
                if fruit_name == "Apples":
                    labels.append(1)
                    Apples += 1
                elif fruit_name == "Banana":
                    labels.append(2)
                    Banana += 1
                elif fruit_name == "Grapes":
                    labels.append(3)
                    Grapes += 1
                elif fruit_name == "Mango":
                    labels.append(4)
                    Mango += 1
                elif fruit_name == "Strawberry":
                    labels.append(5)
                    Strawberry += 1

            except Exception as e:
                print(e)

labels = np.array(labels) - 1
images_train = np.array(images_train, dtype='float32') / 255.0
print(images_train.shape)

# Print class-wise counts
for i, class_name in enumerate(class_names):
    count = np.sum(labels == i)
    print(f"Number of {class_name} : {count}")

print("Total Images: ", len(labels))

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


'''
# Data augmentation using Albumentations
train_transform = A.Compose([
    A.HorizontalFlip(),
    A.Rotate(limit=20),
    A.Normalize(),
    ToTensorV2(),
])# Define the Residual Block

# Data augmentation using torchvision transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
'''
x_train, x_test, y_train, y_test = train_test_split(images_train, labels, random_state=5, shuffle=True, test_size=0.3)

# Prepare the data
x_train_tensor = torch.from_numpy(x_train.transpose(0, 3, 1, 2))
y_train_tensor = torch.from_numpy(y_train)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

x_test_tensor = torch.from_numpy(x_test.transpose(0, 3, 1, 2))
y_test_tensor = torch.from_numpy(y_test)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        # Shortcut connection
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut connection
        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50, self).__init__()

        resnet50_pretrained = models.resnet50(pretrained=pretrained)
        # Adjust the number of input channels in the first convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet50_pretrained.bn1
        self.relu = resnet50_pretrained.relu
        self.maxpool = resnet50_pretrained.maxpool

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = resnet50_pretrained.avgpool
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride=1):
        layers = [ResidualBlock(256, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels * 4, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    


# Load pre-trained ResNet50 model
resnet50_model = models.resnet50(pretrained=True)

# Modify the final fully connected layer
num_classes = 5  # Replace with the number of classes in your dataset
num_features = resnet50_model.fc.in_features
resnet50_model.fc = nn.Linear(num_features, num_classes)

# Print the modified model architecture
print(resnet50_model)


# Prepare for training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50_model.parameters(), lr=0.001, momentum=0.9)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50_model.to(device)

scaler = GradScaler()

for epoch in range(20):
    resnet50_model.train()
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = resnet50_model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    train_accuracy = correct_train / total_train
    print(f'Training Accuracy for Epoch {epoch + 1}: {train_accuracy:.4f}')

# Save the trained model
torch.save(resnet50_model.state_dict(), "resnet_model.pth")

# Evaluate the model
resnet50_model.eval()
predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet50_model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

accuracy = (np.array(predictions) == y_test).mean()
print("Test Accuracy: {:.2f}%".format(accuracy * 100))


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create a confusion matrix (optional)
cm = confusion_matrix(y_test, predictions)
cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)
cm_display.plot()



# Create a confusion matrix with seaborn
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# -------------------------------------------------------------------------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# -------------------------------------------------------------------------------------------------------------------



#                <***************************>      Test Model     <**********************************>  csv result 
import os
import cv2
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision.models as models

# Load the trained model
model = models.resnet50(pretrained=True)
num_classes = 5
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Load the trained weights (replace 'resnet_model.pth' with the actual path)
model.load_state_dict(torch.load("resnet_model.pth"))
model.eval()

# Testing
image_name = []
images_test = []

# Assuming '/path/to/test/' is the path to the directory containing test images
test_path = "/kaggle/input/dataset-n/dataset NN'23/dataset/test"

for img in os.listdir(test_path):
    img_path = os.path.join(test_path, img)

    try:
        image = cv2.imread(img_path)
        if image is None:
            raise Exception(f"Error reading image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)

        images_test.append(image)
        image_name.append(img)

    except Exception as e:
        print(e)

images_test = np.array(images_test, dtype='float32') / 255.0

# Prepare the data
x_test_tensor = torch.from_numpy(images_test.transpose(0, 3, 1, 2))

# Perform predictions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_test_tensor = x_test_tensor.to(device)

model.to(device)
model.eval()

with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs.data, 1)

# Extract image numbers without extension
image_numbers = [int(img.split('.')[0]) for img in image_name]

# Add 1 to the predicted labels if you want to align with the original indexing (1 to 5)
predict_label = predicted.cpu().numpy() + 1

# Save predictions to a CSV file without an index
dict_target = {"image_id": image_numbers, "label": predict_label}
df = pd.DataFrame(dict_target)

# Save the CSV file
csv_filename = "Rez_Net_50_Pre.csv"
df.to_csv(csv_filename, index=False)

# Provide a downloadable link
print(f"CSV file saved as {csv_filename}")
print(f"Download link: [Download CSV](sandbox:/path/to/{csv_filename})")

import os

current_directory = os.getcwd()
print("Current Directory:", current_directory)


df.to_csv("/kaggle/working/Rez_Net_50_Pre.csv", index=False)
import pandas as pd

# Load the CSV file
df_predictions = pd.read_csv("Rez_Net_50_Pre.csv")

# Display the contents of the DataFrame
display(df_predictions)

# Provide a download link

from IPython.display import FileLink

# Provide a download link for the CSV file
FileLink(r'Rez_Net_50_Pre.csv')


# In[ ]:


# Transformer Model :>>>>>>>


# In[ ]:


#####################################   ((( Transformer Model )))    #############################################################


import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_addons.layers import MultiHeadAttention

# Preprocessing and loading data
class_names = ["Apples", "Banana", "Grapes", "Mango", "Strawberry"]
path = "/kaggle/input/dataset-n/dataset NN'23/dataset/train"

images_train = []
labels = []

image_size = (224, 224)

Apples = 0
Banana = 0
Grapes = 0
Mango = 0
Strawberry = 0

for subdir in os.listdir(path):
    subdir_path = os.path.join(path, subdir)
    if os.path.isdir(subdir_path):
        for img in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img)

            try:
                image = cv2.imread(img_path)
                if image is None:
                    raise Exception(f"Error reading image: {img_path}")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size)

                images_train.append(image)

                fruit_name = class_names[int(subdir) - 1]
                if fruit_name == "Apples":
                    labels.append(1)
                    Apples += 1
                elif fruit_name == "Banana":
                    labels.append(2)
                    Banana += 1
                elif fruit_name == "Grapes":
                    labels.append(3)
                    Grapes += 1
                elif fruit_name == "Mango":
                    labels.append(4)
                    Mango += 1
                elif fruit_name == "Strawberry":
                    labels.append(5)
                    Strawberry += 1

            except Exception as e:
                print(e)

labels = np.array(labels) - 1
images_train = np.array(images_train, dtype='float32') / 255.0
print(images_train.shape)

x_train, x_test, y_train, y_test = train_test_split(images_train, labels, random_state=5, shuffle=True, test_size=0.3)

# Vision Transformer (ViT) model:
def create_vit_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu")(inputs)
    x = layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu")(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Reshape((1, 512))(x)

    # Vision Transformer (ViT) architecture
    transformer_block = tf.keras.layers.MultiHeadAttention(
        num_heads=8, key_dim=512 // 8, dropout=0.1
    )
    x = transformer_block(x, x)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

# Create ViT model
input_shape = (224, 224, 3)
num_classes = 5
vit_model = create_vit_model(input_shape, num_classes)

# Compile the model
vit_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
vit_model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.1)

# Save the trained model
vit_model.save("vit_model.h5")
print("_________________________________________________________________________")
# Evaluate the model
test_loss, test_accuracy = vit_model.evaluate(x_test, y_test)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
print("_________________________________________________________________________")
# Create confusion matrix (optional)
predictions = np.argmax(vit_model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, predictions)
cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)
cm_display.plot()

# -------------------------------------------------------------------------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# -------------------------------------------------------------------------------------------------------------------


#                <***************************>      Test Model     <**********************************>  csv result 

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models

# Load the ViT model
vit_model = models.load_model("vit_model.h5")

# Testing
class_names = ["Apples", "Banana", "Grapes", "Mango", "Strawberry"]

image_name = []
images_test = []

# Assuming '/path/to/test/' is the path to the directory containing test images
test_path = "/kaggle/input/dataset-n/dataset NN'23/dataset/test"

for img in os.listdir(test_path):
    img_path = os.path.join(test_path, img)

    try:
        image = cv2.imread(img_path)
        if image is None:
            raise Exception(f"Error reading image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  # Assuming input shape of your ViT model is (224, 224, 3)

        images_test.append(image)
        image_name.append(img)

    except Exception as e:
        print(e)

images_test = np.array(images_test, dtype='float32') / 255.0

# Perform predictions
predictions = np.argmax(vit_model.predict(images_test), axis=1)

# Extract image numbers without extension
image_numbers = [int(img.split('.')[0]) for img in image_name]

# Add 1 to the predicted labels if you want to align with the original indexing (1 to 5)
predict_label = predictions + 1

# Save predictions to a CSV file without an index
dict_target = {"image_id": image_numbers, "label": predict_label}
df = pd.DataFrame(dict_target)

# Save the CSV file
csv_filename = "Vision_Transformer.csv"
df.to_csv(csv_filename, index=False)

# Provide a downloadable link
print(f"CSV file saved as {csv_filename}")
print(f"Download link: [Download CSV](sandbox:/path/to/{csv_filename})")

import os

current_directory = os.getcwd()
print("Current Directory:", current_directory)


df.to_csv("/kaggle/working/Vision_Transformer.csv", index=False)
import pandas as pd

# Load the CSV file
df_predictions = pd.read_csv("Vision_Transformer.csv")

# Display the contents of the DataFrame
display(df_predictions)

# Provide a download link
from IPython.display import FileLink

# Provide a download link for the CSV file
FileLink(r'Vision_Transformer.csv')


# In[ ]:


# new modified Transformer model :


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Dense, GlobalAveragePooling2D, Reshape
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.layers import MultiHeadAttention
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def load_training_data(data_folder, num_classes, image_size):
    train_samples = []
    train_labels = []

    for class_label in range(1, num_classes + 1):
        class_path = os.path.join(data_folder, str(class_label))
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img, image_size)
            train_samples.append(img_resized)
            train_labels.append(class_label - 1)

    train_samples, train_labels = np.array(train_samples), np.array(train_labels)

    return train_samples, train_labels

def calculate_pairwise_distances(embeddings):
    distances = pairwise_distances(embeddings, metric='euclidean')
    return distances

def patchify(image, patch_size):
    # Calculate the number of patches in each dimension
    num_patches_x = image.shape[1] // patch_size
    num_patches_y = image.shape[2] // patch_size

    # Reshape the image to facilitate patch extraction
    reshaped_image = tf.reshape(image, [-1, num_patches_x, patch_size, num_patches_y, patch_size, image.shape[-1]])

    # Swap the dimensions to get patches in the correct order
    patches = tf.transpose(reshaped_image, [0, 1, 3, 2, 4, 5])

    # Reshape to the final shape
    patches = tf.reshape(patches, [-1, patch_size, patch_size, image.shape[-1]])

    return patches, num_patches_x * num_patches_y



def vision_transformer_model(input_shape, num_classes, patch_size=4, embedding_dim=256, num_heads=2, head_size=128, dropout=0.1):
   # Input layer
    input_tensor = Input(shape=input_shape)
    
    # Entry layers
    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Additional Convolutional layers
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Reshape to prepare for patch extraction
    x = Reshape((-1, 128))(x)
    
    # Extract patches from the image
    patches, num_patches = patchify(x, patch_size)
    
    # Flatten the image patches
    flattened_patches = tf.keras.layers.Flatten()(patches)
    
    # Linear embeddings
    embeddings = Dense(embedding_dim)(flattened_patches)
    
    # Positional Encoding
    positional_encoding = tf.keras.layers.PositionalEncoding()(tf.zeros_like(embeddings))
    
    # Add positional encoding to embeddings
    x = embeddings + positional_encoding
    
    # Reshape to prepare for MultiHeadAttention layer
    x = Reshape((num_patches, -1))(x)
    
    # Vision Transformer (ViT) layers
    # Encoding part begins here
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    # Encoding part ends here
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    
    # Additional Dense layer
    x = Dense(128, activation='relu')(x)
    
    # Output layer
    output_tensor = Dense(num_classes, activation='softmax')(x)

    # Model instantiation
    model = Model(inputs=input_tensor, outputs=output_tensor)
    
    return model



# Load and preprocess data
image_size = (64, 64)
num_classes = 5
data_folder = "/kaggle/input/dataset-n/dataset NN'23/dataset/train"
train_samples, train_labels = load_training_data(data_folder, num_classes, image_size)

# Split data into training and validation sets
train_samples, val_samples, train_labels, val_labels = train_test_split(
    train_samples, train_labels, test_size=0.2, random_state=42
)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(train_samples)

# Build and compile the vision transformer model
input_shape = (image_size[0], image_size[1], 3)
vit_model = vision_transformer_model(input_shape, num_classes)
vit_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
vit_model.summary()

# Train the vision transformer model with data augmentation
history = vit_model.fit(
    datagen.flow(train_samples, train_labels, batch_size=128),
    epochs=200,
    validation_data=(val_samples, val_labels),
    batch_size=256
)




# Print overall training accuracy
train_accuracy = vit_model.evaluate(train_samples, train_labels, verbose=0)[1]
print(f"Overall Accuracy: {train_accuracy * 100:.2f}%")

# Save the vision transformer model
model_save_path = '/kaggle/working/vision_transformer_model.h5'
vit_model.save(model_save_path)
print(f"Vision Transformer model saved to: {model_save_path}")

# Now you can use the encoder for encoding and make predictions with the trained model.
# For example, to encode an image:
encoded_representation = vit_model.predict(np.expand_dims(train_samples[0], axis=0))

# To make predictions:
predictions = vit_model.predict(np.expand_dims(val_samples[0], axis=0))

# Confusion Matrix
predictions = vit_model.predict(val_samples)
conf_matrix = confusion_matrix(val_labels, np.argmax(predictions, axis=1))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=range(num_classes))
disp.plot(cmap='viridis', values_format='d')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:


# EfficientNet Scratch only :>>>>>> ((final))


# In[ ]:


########################-------------->>>>>>>>>            EfficientNet Scratch only        <<<<<<<<<<______________#################################

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.cuda.amp import autocast, GradScaler
from torchsummary import summary


# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# datagen.fit(train_samples)

# Updated CNNBlock and InvertedResidualBlock implementations
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, padding):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x + identity

# Custom dataset class for preprocessing and loading data
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Preprocessing and loading data
class_names = ["Apples", "Banana", "Grapes", "Mango", "Strawberry"]
path = "/kaggle/input/mosaleh/dataset NN'23/dataset/train"

# Loading and preprocessing images
images_train = []
labels = []

image_size = (224, 224)

Apples = 0
Banana = 0
Grapes = 0
Mango = 0
Strawberry = 0

for subdir in os.listdir(path):
    subdir_path = os.path.join(path, subdir)
    if os.path.isdir(subdir_path):
        for img in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img)

            try:
                image = cv2.imread(img_path)
                if image is None:
                    raise Exception(f"Error reading image: {img_path}")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size)

                images_train.append(image)

                fruit_name = class_names[int(subdir) - 1]
                if fruit_name == "Apples":
                    labels.append(1)  # Use 0-based indexing for labels
                    Apples += 1
                elif fruit_name == "Banana":
                    labels.append(2)
                    Banana += 1
                elif fruit_name == "Grapes":
                    labels.append(3)
                    Grapes += 1
                elif fruit_name == "Mango":
                    labels.append(4)
                    Mango += 1
                elif fruit_name == "Strawberry":
                    labels.append(5)
                    Strawberry += 1

            except Exception as e:
                print(e)

labels = np.array(labels) - 1
images_train = np.array(images_train, dtype='float32') / 255.0
print(images_train.shape)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(images_train, labels, random_state=5, shuffle=True, test_size=0.3)

# Transformations (you can customize these based on your needs)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = CustomDataset(images=x_train, labels=y_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# # Create an instance of the model
# model = EfficientNet.from_name('efficientnet-b0', num_classes=5)  # Adjust num_classes based on your problem
# print(model)
# Create an instance of the model with pre-trained weights
model = EfficientNet.from_name('efficientnet-b0', num_classes=5)  # Adjust num_classes based on your problem
# print(model)
# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

scaler = GradScaler()

for epoch in range(50):  # Adjust the number of epochs as needed
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), "efficientnet_model.pth")
# Evaluation
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)
cm_display.plot()


# -------------------------------------------------------------------------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# -------------------------------------------------------------------------------------------------------------------


#                <***************************>      Test Model     <**********************************>  csv result 


import pandas as pd

# Path to the directory containing test images
test_path = "/kaggle/input/mosaleh/dataset NN'23/dataset/test"

# Create lists to store image IDs and predicted labels
image_ids = []
predicted_labels = []

# Load the trained model
model = EfficientNet.from_name('efficientnet-b0', num_classes=5)
model.load_state_dict(torch.load("efficientnet_model.pth"))
model.to(device)
model.eval()

# Iterate through test images
for img_file in os.listdir(test_path):
    img_path = os.path.join(test_path, img_file)

    try:
        image = cv2.imread(img_path)
        if image is None:
            raise Exception(f"Error reading image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted_label = torch.max(output, 1)

        # Append results to lists
        image_ids.append(int(img_file.split('.')[0]))
        predicted_labels.append(int(predicted_label.item()) + 1)

    except Exception as e:
        print(e)

# Create a DataFrame from lists
results_df = pd.DataFrame({"image_id": image_ids, "label": predicted_labels})

# Save the results to a CSV file
results_df.to_csv("No_Pre_EfficientNet.csv", index=False)

# Print the contents of the DataFrame
print(results_df)



# Create a download link
from IPython.display import FileLink

FileLink(r'No_Pre_EfficientNet.csv')


# In[ ]:


# EfficientNet_Pretrain & Scratch :>>>>>> ((final))


# In[ ]:


########################-------------->>>>>>>>>            EfficientNet_Pretrain & Scratch         <<<<<<<<<<____________#################################

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.cuda.amp import autocast, GradScaler
from torchsummary import summary

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(train_samples)

# Updated CNNBlock and InvertedResidualBlock implementations
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, padding):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x + identity

# Custom dataset class for preprocessing and loading data
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Preprocessing and loading data
class_names = ["Apples", "Banana", "Grapes", "Mango", "Strawberry"]
path = "/kaggle/input/mosaleh/dataset NN'23/dataset/train"

# Loading and preprocessing images
images_train = []
labels = []

image_size = (224, 224)

Apples = 0
Banana = 0
Grapes = 0
Mango = 0
Strawberry = 0

for subdir in os.listdir(path):
    subdir_path = os.path.join(path, subdir)
    if os.path.isdir(subdir_path):
        for img in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img)

            try:
                image = cv2.imread(img_path)
                if image is None:
                    raise Exception(f"Error reading image: {img_path}")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size)

                images_train.append(image)

                fruit_name = class_names[int(subdir) - 1]
                if fruit_name == "Apples":
                    labels.append(1)  # Use 0-based indexing for labels
                    Apples += 1
                elif fruit_name == "Banana":
                    labels.append(2)
                    Banana += 1
                elif fruit_name == "Grapes":
                    labels.append(3)
                    Grapes += 1
                elif fruit_name == "Mango":
                    labels.append(4)
                    Mango += 1
                elif fruit_name == "Strawberry":
                    labels.append(5)
                    Strawberry += 1

            except Exception as e:
                print(e)

labels = np.array(labels) - 1
images_train = np.array(images_train, dtype='float32') / 255.0
print(images_train.shape)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(images_train, labels, random_state=5, shuffle=True, test_size=0.3)

# Transformations (you can customize these based on your needs)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = CustomDataset(images=x_train, labels=y_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# # Create an instance of the model
# model = EfficientNet.from_name('efficientnet-b0', num_classes=5)  # Adjust num_classes based on your problem
# print(model)
# Create an instance of the model with pre-trained weights
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)  # Adjust num_classes based on your problem
# print(model)
# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

scaler = GradScaler()

for epoch in range(50):  # Adjust the number of epochs as needed
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), "efficientnet_model.pth")
# Evaluation
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)
cm_display.plot()


# -------------------------------------------------------------------------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# -------------------------------------------------------------------------------------------------------------------


#                <***************************>      Test Model     <**********************************>  csv result 


import pandas as pd

# Path to the directory containing test images
test_path = "/kaggle/input/mosaleh/dataset NN'23/dataset/test"

# Create lists to store image IDs and predicted labels
image_ids = []
predicted_labels = []

# Load the trained model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
model.load_state_dict(torch.load("efficientnet_model.pth"))
model.to(device)
model.eval()

# Iterate through test images
for img_file in os.listdir(test_path):
    img_path = os.path.join(test_path, img_file)

    try:
        image = cv2.imread(img_path)
        if image is None:
            raise Exception(f"Error reading image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted_label = torch.max(output, 1)

        # Append results to lists
        image_ids.append(int(img_file.split('.')[0]))
        predicted_labels.append(int(predicted_label.item()) + 1)

    except Exception as e:
        print(e)

# Create a DataFrame from lists
results_df = pd.DataFrame({"image_id": image_ids, "label": predicted_labels})

# Save the results to a CSV file
results_df.to_csv("Pre_EfficientNet.csv", index=False)

# Print the contents of the DataFrame
print(results_df)



# Create a download link
from IPython.display import FileLink

FileLink(r'Pre_EfficientNet.csv')


# In[ ]:


# 


# In[ ]:


# #######################      EfficientNet_Pre_bulit_in & Scratch  #################################


# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import transforms
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from efficientnet_pytorch import EfficientNet
# from torch.cuda.amp import autocast, GradScaler

# # Preprocessing and loading data
# class_names = ["Apples", "Banana", "Grapes", "Mango", "Strawberry"]
# path = "/kaggle/input/mosaleh/dataset NN'23/dataset/train"

# images_train = []
# labels = []

# image_size = (224, 224)

# Apples = 0
# Banana = 0
# Grapes = 0
# Mango = 0
# Strawberry = 0

# for subdir in os.listdir(path):
#     subdir_path = os.path.join(path, subdir)
#     if os.path.isdir(subdir_path):
#         for img in os.listdir(subdir_path):
#             img_path = os.path.join(subdir_path, img)

#             try:
#                 image = cv2.imread(img_path)
#                 if image is None:
#                     raise Exception(f"Error reading image: {img_path}")

#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = cv2.resize(image, image_size)

#                 images_train.append(image)

#                 fruit_name = class_names[int(subdir) - 1]
#                 if fruit_name == "Apples":
#                     labels.append(1)
#                     Apples += 1
#                 elif fruit_name == "Banana":
#                     labels.append(2)
#                     Banana += 1
#                 elif fruit_name == "Grapes":
#                     labels.append(3)
#                     Grapes += 1
#                 elif fruit_name == "Mango":
#                     labels.append(4)
#                     Mango += 1
#                 elif fruit_name == "Strawberry":
#                     labels.append(5)
#                     Strawberry += 1

#             except Exception as e:
#                 print(e)

# labels = np.array(labels) - 1
# images_train = np.array(images_train, dtype='float32') / 255.0
# print(images_train.shape)

# x_train, x_test, y_train, y_test = train_test_split(images_train, labels, random_state=5, shuffle=True, test_size=0.3)

# # Step 1: Instantiate the model with pre-trained weights
# class ModifiedEfficientNet(nn.Module):
#     def __init__(self, num_classes=5, pretrained=True):
#         super(ModifiedEfficientNet, self).__init__()
#         if pretrained:
#             self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
#         else:
#             self.base_model = EfficientNet.from_name('efficientnet-b0')

#         self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_classes)

#     def forward(self, x):
#         return self.base_model(x)

# model = ModifiedEfficientNet(num_classes=len(class_names), pretrained=True)

# # Step 2: Prepare the data with transformations
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(degrees=(-0.2, 0.2)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

    
# train_dataset = CustomDataset(x_train, y_train, transform=transform)
# test_dataset = CustomDataset(x_test, y_test, transform=transforms.ToTensor())

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Step 3: Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Step 4: Train the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# scaler = GradScaler()

# for epoch in range(50):
#     model.train()
#     correct_train = 0
#     total_train = 0
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         _, predicted_train = torch.max(outputs.data, 1)
#         total_train += labels.size(0)
#         correct_train += (predicted_train == labels).sum().item()
# # ...

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, images, labels, transform=None):
#         self.images = images
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         image = self.images[index]
#         label = self.labels[index]

#         # Ensure the image has three channels (RGB)
#         if image.shape[-1] > 3:
#             image = image[:, :, :3]

#         # Convert to float32 if not already
#         if image.dtype != np.float32:
#             image = image.astype(np.float32) / 255.0

#         # Convert to torch tensor if not already
#         if not isinstance(image, torch.Tensor):
#             image = torch.from_numpy(image)

#         # Convert to PIL Image
#         image = transforms.ToPILImage()(image)

#         if self.transform:
#             image = self.transform(image)

#         return image, label

# # ...

#     train_accuracy = correct_train / total_train
#     print(f'Training Accuracy for Epoch {epoch + 1}: {train_accuracy:.4f}')

# # Step 5: Save the trained model
# torch.save(model.state_dict(), "modified_efficientnet_model.pth")

# # Step 6: Evaluate the model
# model.eval()
# predictions = []
# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         predictions.extend(predicted.cpu().numpy())

# accuracy = (np.array(predictions) == y_test).mean()
# print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# # Step 7: Create confusion matrix (optional)
# cm = confusion_matrix(y_test, predictions)
# cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)
# cm_display.plot()



# # -------------------------------------------------------------------------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # -------------------------------------------------------------------------------------------------------------------


# #                <***************************>      Test Model     <**********************************>  csv result 



# import pandas as pd

# # Path to the directory containing test images
# test_path = "/kaggle/input/mosaleh/dataset NN'23/dataset/test"

# # Create lists to store image IDs and predicted labels
# image_ids = []
# predicted_labels = []

# # Load the trained model
# model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
# model.load_state_dict(torch.load("modified_efficientnet_model.pth"))
# model.to(device)
# model.eval()

# # Iterate through test images
# for img_file in os.listdir(test_path):
#     img_path = os.path.join(test_path, img_file)

#     try:
#         image = cv2.imread(img_path)
#         if image is None:
#             raise Exception(f"Error reading image: {img_path}")

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, image_size)
#         image = transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = model(image)
#             _, predicted_label = torch.max(output, 1)

#         # Append results to lists
#         image_ids.append(int(img_file.split('.')[0]))
#         predicted_labels.append(int(predicted_label.item()) + 1)

#     except Exception as e:
#         print(e)

# # Create a DataFrame from lists
# results_df = pd.DataFrame({"image_id": image_ids, "label": predicted_labels})

# # Save the results to a CSV file
# results_df.to_csv("NoN_EfficientNet.csv", index=False)

# # Print the contents of the DataFrame
# print(results_df)

# # Create a download link
# from IPython.display import FileLink

# FileLink(r'NoN_EfficientNet.csv')



# In[ ]:




