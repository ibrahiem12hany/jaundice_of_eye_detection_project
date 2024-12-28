import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dataset paths
data_dir = r"D:\\5-Deep Learning\\jaundiced_normal"



# Enhance function to highlight jaundice
def enhance_image_for_jaundice(image):
    image_hsv = image.convert('HSV')        # تحويل الصورة إلى نمط HSV لاستخراج اللون الأصفر
    h, s, v = image_hsv.split()
    # يمكن تحديد نطاق اللون الأصفر بناءً على التدرج اللوني. اللون الأصفر عادةً يحتوي على قيمة H بين 20-60
    h = h.point(lambda p: p if 20 < p < 60 else p)
    image_hsv = Image.merge('HSV', (h, s, v))
    image = image_hsv.convert('RGB')
    enhancer = ImageEnhance.Color(image)    # تعزيز الألوان الأخرى
    image = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    return image


# Data transformations with enhancement
transform = transforms.Compose([
    transforms.Lambda(lambda x: enhance_image_for_jaundice(x)),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Loading dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)




# Define CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Model selection
model_choice = "CNN"  # Change to "ResNet" for ResNet18
if model_choice == "CNN":
    model = CNNModel(num_classes=len(class_names)).to(device)
else:
    pretrained_model = models.resnet18(pretrained=True)
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, len(class_names))
    model = pretrained_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train the model
def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()    
            # حساب الدقة
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")




    # رسم منحنيات الخسارة والدقة
    plot_training_metrics(train_losses, train_accuracies)
def plot_training_metrics(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))
    # رسم الخسارة
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # رسم الدقة
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()



# Save the model
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

# Evaluate the model and return labels and predictions
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return all_labels, all_preds




# Confusion matrix visualization
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
# Visualize predictions
def visualize_predictions(model, test_loader, num_images=5):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    # تحويل الصور إلى شكل مناسب للعرض
    images = images[:num_images]
    labels = labels[:num_images]
    # التنبؤات
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)
    # عرض الصور مع التنبؤات
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).cpu().numpy()  # تحويل التنسور إلى صورة قابلة للعرض
        img = (img - img.min()) / (img.max() - img.min())  # تطبيع الصورة
        ax.imshow(img)
        ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        ax.axis('off')
    plt.show()



# Train the model and save
train_model(model, criterion, optimizer, train_loader, num_epochs=10)
save_model(model, r"D:\\5-Deep Learning\\jaundiced_normal\\trained_model.pth")

# Evaluate model and plot confusion matrix
all_labels, all_preds = evaluate_model(model, test_loader)
plot_confusion_matrix(all_labels, all_preds)

# Visualize some predictions
visualize_predictions(model, test_loader)



# GUI for image prediction
class_names = ['Jaundiced Eyes', 'Normal Eyes']  # Ensure the class names match the model

def load_model_for_gui():
    if model_choice == "CNN":
        model = CNNModel(num_classes=len(class_names))
    else:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(r"D:\\5-Deep Learning\\jaundiced_normal\\trained_model.pth"))
    model.to(device)
    model.eval()
    return model

# Function to process the image for prediction
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Prediction function
def predict_image(model, image):
    image = image.to(device)
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return preds.item()

# Upload and predict
def upload_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((300, 300))  # Resize for display
        img_display = ImageTk.PhotoImage(image)
        img_label.config(image=img_display)
        img_label.image = img_display

        # Process and predict
        image_tensor = process_image(file_path)
        prediction = predict_image(model, image_tensor)

        predicted_class = class_names[prediction]
        result_label.config(text=f"Prediction: {predicted_class}")
    else:
        messagebox.showwarning("No image selected", "Please select an image to predict.")

# Initialize the Tkinter root window
root = tk.Tk()
root.title("Jaundice Prediction")
root.geometry("400x500")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

img_label = tk.Label(root)
img_label.pack(pady=10)

result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14))
result_label.pack(pady=20)

root.mainloop()
