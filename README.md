# Transformer Model Training with HiTZ/latxa-7b-v1

This repository contains scripts and resources for training a transformer model using the `HiTZ/latxa-7b-v1` architecture, tailored for high-performance language understanding. The model is trained using the Oxford English dictionary corpus.

## Repository Structure

- **Training/**:
  - Contains the main training script `eeh_training.py` to train the model on NVIDIA 2x3090 GPUs.
  - **Data/**: Stores training and validation datasets generated by the training script.
  - **Results/**: Includes the saved model checkpoints after training.

## Corpora

- External corpora used for training can be accessed through the following link:
  - [Basque Corpus](https://github.com/eneko98/EEH-Corpus.git)

## Model Training

The model utilizes the `HiTZ/latxa-7b-v1` transformer architecture from HuggingFace. Training is performed on NVIDIA 2x3090 GPUs. The training process emphasizes efficient learning and memory management to handle the extensive dataset provided by the Oxford Corpus. Key training enhancements include:

- **QLora**: Quantized Layers for Reduced memory.
- **Peft**: Progressive layer freezing for efficiency.
- **EarlyStopping**: To prevent overfitting and optimize training time.

## Model Evaluation

To test the trained model, use the `eeh_evaluation.py` script in the `Evaluation` folder. This script evaluates the model's performance on generating precise and contextually accurate definitions from the Oxford Corpus.

## Setup and Usage

To set up the training environment and run the training scripts, follow these instructions:

1. **Clone the Repository:**
```
git clone https://github.com/eneko98/Latxa-Basque.git
```
```
cd Latxa-Basque
```

2. **Install Dependencies:**
```
pip install -r requirements.txt
```

3. **Run Training Script:**
```
python eeh_training.py
```

4. **Evaluate the Model:**
```
python eeh_evaluation.py
```

5. **Contributing:**
Contributions to this project are welcome. Please fork the repository and submit a pull request to propose changes.
