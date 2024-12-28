Jaundice Detection Model with GUI
Overview:
This code implements a deep learning model for detecting jaundice in images of eyes, where the model classifies images into two categories: "Jaundiced Eyes" and "Normal Eyes." The model is based on convolutional neural networks (CNN), ResNet and are integrated with a graphical user interface (GUI) to allow users to upload images for prediction.




Running the Code:
Libraries Used:
•	torch: PyTorch for deep learning tasks.
•	torchvision: Provides models and image transformations.
•	matplotlib: Used for plotting graphs.
•	sklearn.metrics: For evaluating the model using confusion matrix.
•	seaborn: For better visualization of confusion matrix.
•	tkinter: For building the GUI application.
•	PIL: For image processing (resizing, enhancing, etc.).
To install them ‘’pip install torch torchvision matplotlib scikit-learn seaborn tk pillow’’ on CMD
Download the Dataset: https://www.kaggle.com/datasets/puspendrakumar77/jaundiced-and-normal-eyes
Run the Script:
•	Open the Python script containing the code.
•	When prompted, type your choice for the model by model_choice = "CNN"  or "ResNet"
•	Execute the script to start the training process and the GUI interface.
If you want to train the model, the script will automatically train for 10 epochs. After training, it saves the trained model. Once trained, the model will be ready for evaluation and prediction.
For prediction, the GUI will open, allowing you to upload an image and get a prediction of whether the image contains "Jaundiced Eyes" or "Normal Eyes".

Key Sections:
1. Device Setup:
The code checks if a GPU (CUDA) is available and sets the device accordingly:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
2. Image Enhancement Function:
The enhance_image_for_jaundice function enhances the image to highlight jaundice by manipulating the color and contrast.
3. Data Preprocessing:
Transforms are applied to images, which include resizing, random horizontal flips, random rotations, and the enhancement function.
transform = transforms.Compose([
    transforms.Lambda(lambda x: enhance_image_for_jaundice(x)),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
4. Loading Dataset:
The dataset is loaded from a specified directory, split into training and test sets, and data loaders are created:
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
5. CNN and ResNet Model Definition:
A custom CNN model is defined, or alternatively, a pre-trained ResNet18 model is used for transfer learning:
class CNNModel(nn.Module):
Alternatively, the ResNet model can be loaded as follows:
pretrained_model = models.resnet18(pretrained=True)
pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, len(class_names))
6. Training the Model:
The model is trained over multiple epochs, and metrics like loss and accuracy are calculated:
def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
Loss and accuracy plots are generated for visual analysis:
plot_training_metrics(train_losses, train_accuracies)
7. Saving the Model:
Once the model is trained, it is saved to a file for future use:
torch.save(model.state_dict(), filepath)
8. Evaluation:
The model is evaluated on the test set, and accuracy along with a confusion matrix is generated:
def evaluate_model(model, test_loader):
The confusion matrix is visualized using seaborn:
def plot_confusion_matrix(labels, preds):
9. GUI for Image Prediction:
A GUI is created using Tkinter, which allows users to upload an image and get a prediction:
The prediction result is displayed on the GUI after processing the image and passing it through the trained model.
Functions:
•	enhance_image_for_jaundice: Enhances images to highlight yellow hues associated with jaundice.
•	train_model: Trains the model on the dataset.
•	plot_training_metrics: Visualizes training loss and accuracy.
•	save_model: Saves the trained model to disk.
•	evaluate_model: Evaluates the model's performance on the test set.
•	plot_confusion_matrix: Displays a confusion matrix for model evaluation.
•	visualize_predictions: Visualizes predictions on test set images.
•	load_model_for_gui: Loads the trained model for use in the GUI.
•	process_image: Processes an image for prediction by the model.
•	predict_image: Makes a prediction for a single image.
•	upload_image: Opens a file dialog to upload an image for prediction.
GUI Features:
•	Image Upload: Users can upload an image through the file dialog.
•	Display Image: The uploaded image is displayed in the GUI.
•	Prediction Result: The model predicts whether the image contains "Jaundiced Eyes" or "Normal Eyes," and the result is shown in the GUI.
Conclusion:
This code allows for both training a deep learning model for jaundice detection and using a simple GUI to predict the condition from uploaded images.
