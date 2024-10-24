# Fish Species Classification using ResNet50
### This project implements a fish species classification model using transfer learning with a pre-trained ResNet50 model. The dataset consists of various types of fish images, which are pre-processed and classified into different species. The model achieves a high level of accuracy with the help of fine-tuning.

# Dataset
### The dataset used in this project is A Large-Scale Fish Dataset for Segmentation and Classification, which contains labeled images of various fish species.
### Link to the dataset: https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

# Project Structure
## Data Loading and Preprocessing
### The dataset is loaded, and the paths and labels are extracted.
### The images are split into training, validation, and test sets using train_test_split.
### Images are preprocessed using ImageDataGenerator with the MobileNetV2 preprocessing function.
## Model Architecture
### A pre-trained ResNet50 model is used as the base model without its top layers (headless).
### The architecture consists of Global Average Pooling, followed by Dense and Dropout layers to prevent overfitting.
## Training
### The model is trained with a learning rate scheduler and early stopping to optimize performance.
### Initially, the ResNet50 base model is frozen, and only the added layers are trained.
## Fine-Tuning
### After the initial training, the top layers of ResNet50 are unfrozen, and the model is fine-tuned on the specific dataset.
## Evaluation
### The model is evaluated on the test set, achieving strong accuracy.
### A confusion matrix and classification report are generated to analyze performance per class.
## Visualization
### Sample predictions are visualized along with their true labels for better interpretation.
# Results
### The model achieves over 95% accuracy on the test set.
### Detailed performance metrics are provided, including precision, recall, and F1 scores for each class.
# How to Run
### Download the dataset from Kaggle.
### Follow the notebook steps, from loading the data to training and evaluating the model.
### The model can be fine-tuned further for more improvements if needed.
# Conclusion
### This project demonstrates how transfer learning with ResNet50 can be applied to fish species classification, achieving high accuracy and solid generalization performance.
### Link To my Kaggle Notebook: https://www.kaggle.com/code/bartaalan/fish-class-project

# Citations
@inproceedings{ulucan2020large, title={A Large-Scale Dataset for Fish Segmentation and Classification}, author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet}, booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)}, pages={1--5}, year={2020}, organization={IEEE} }

O.Ulucan, D.Karakaya, and M.Turkan.(2020) A large-scale dataset for fish segmentation and classification. In Conf. Innovations Intell. Syst. Appli. (ASYU)
