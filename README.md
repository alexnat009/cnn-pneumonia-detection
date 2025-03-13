# Pneumonia Detection Using CNN  

## Project Overview  
This project implements a **Convolutional Neural Network (CNN)** for detecting pneumonia from chest X-ray images. The dataset used is the **Chest X-ray dataset**, which consists of labeled images categorized as *Normal* or *Pneumonia*.  

## Model Architecture  
The CNN model consists of:  
- **3 Convolutional Layers** (with ReLU activation)  
- **3 Max-Pooling Layers**  
- **Flatten Layer**  
- **Fully Connected Dense Layer** (128 neurons, ReLU activation)  
- **Output Layer** (1 neuron, Sigmoid activation for binary classification)  

The model is trained using the **Adam optimizer** with **binary cross-entropy loss**.  

## Dataset  
The dataset is organized into three folders:  
```
dataset/chest_xray/
│── train/       # Training images
│── val/         # Validation images
│── test/        # Testing images
```

## 🔧 Installation & Setup  
11) Clone the repository:  
```bash
git clone https://github.com/alexnat009/cnn-pneumonia-detection.git

cd cnn-pneumonia-detection
```  
2) Install dependencies:  
```bash
pip install numpy pandas matplotlib tensorflow
```  
3) Run the model training:  
```bash
python main.py
```  

## Model Training & Evaluation  
- The model is trained for **10+2 epochs**  
- Training and validation accuracy/loss are plotted for analysis  
- The model is evaluated on a separate test set  

## Visualization  
The script generates **accuracy & loss curves** to monitor model performance
