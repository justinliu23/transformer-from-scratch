# transformer-from-scratch

## Table of Contents

- [Installation Instructions](#installation-instructions)
  - [Prerequisites](#prerequisites)
  - [Dependencies](#dependencies)
- [Usage](#usage)
  - [Example Usage](#example-usage)
- [Features](#features)
- [Configuration](#configuration)

## Installation Instructions

### Prerequisites

To run this project, ensure you have the following:

- Python 3.7 or later
- Jupyter Notebook (to run the `.ipynb` notebook files)
- An environment with TensorFlow 2.x installed

### Dependencies

You can install all the required dependencies using `pip`. To do so, execute the following command in your terminal:

```bash
pip install tensorflow numpy matplotlib transformers
```

## Usage

To use the Transformer model for NLP tasks, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/justinliu23/transformer-from-scratch.git
   cd transformer-network-project
   ```

2. **Open Jupyter Notebook**:
   Launch Jupyter Notebook in your terminal:

   ```bash
   jupyter notebook
   ```

   Open the notebook file `Transformer.ipynb`.

3. **Run the Cells**:
   Execute the cells sequentially in the Jupyter Notebook to understand the flow of the model construction, training, and evaluation.

### Example Usage

Here is an example workflow to use the model:

1. **Positional Encoding**: The notebook first demonstrates the implementation of positional encoding, which helps the model understand the order of words in a sentence.

2. **Self-Attention Calculation**: The notebook shows how to compute scaled dot-product self-attention scores, which allows the model to focus on different parts of the input sentence.

3. **Masked Multi-Head Attention**: The notebook provides the implementation of masked multi-head attention for capturing different aspects of the input sequence concurrently.

4. **Build and Train the Transformer Model**: Finally, the notebook builds a full Transformer model and trains it on a sample dataset.

## Features

- **Positional Encoding**: Implements sinusoidal positional encodings to provide the model with information about the relative positions of words in a sentence.
- **Self-Attention Mechanism**: Efficiently computes self-attention scores to help the model focus on important parts of the input sequence.
- **Masked Multi-Head Attention**: Allows the model to attend to different parts of the input sequence simultaneously.
- **Transformer Model Construction**: Builds a full Transformer model using TensorFlow/Keras, incorporating all the elements of the Transformer architecture.
- **Model Training and Evaluation**: Trains the Transformer model on a sample dataset and evaluates its performance.

## Configuration

The notebook is designed to run with the default settings, but you may need to configure certain parameters based on your specific use case:

- **Model Hyperparameters**: You can adjust parameters such as the number of layers, attention heads, and embedding dimensions in the notebook cells where the Transformer model is constructed.
- **Training Parameters**: Modify the learning rate, batch size, and number of epochs in the training section to optimize the model's performance for your dataset.
- **Data Preprocessing**: Ensure that your input data is preprocessed correctly. The notebook uses tokenization and padding techniques from the Hugging Face `transformers` library. You can modify these methods based on your input data's format.
# Transformer Network Project

Welcome to the Transformer Network Project, which implements a Transformer architecture for natural language processing (NLP) tasks. This project demonstrates key concepts such as positional encoding, self-attention, and masked multi-head attention, culminating in the construction of a Transformer model using TensorFlow and Keras.

## Table of Contents

- [Installation Instructions](#installation-instructions)
  - [Prerequisites](#prerequisites)
  - [Dependencies](#dependencies)
- [Usage](#usage)
  - [Example Usage](#example-usage)
- [Features](#features)
- [Configuration](#configuration)

## Installation Instructions

### Prerequisites

To run this project, ensure you have the following:

- Python 3.7 or later
- Jupyter Notebook (to run the `.ipynb` notebook files)
- An environment with TensorFlow 2.x installed

### Dependencies

You can install all the required dependencies using `pip`. To do so, execute the following command in your terminal:

```bash
pip install tensorflow numpy matplotlib transformers
```

## Usage

To use the Transformer model for NLP tasks, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/justinliu23/transformer-from-scratch.git
   cd transformer-network-project
   ```

2. **Open Jupyter Notebook**:
   Launch Jupyter Notebook in your terminal:

   ```bash
   jupyter notebook
   ```

   Open the notebook file `Transformer.ipynb`.

3. **Run the Cells**:
   Execute the cells sequentially in the Jupyter Notebook to understand the flow of the model construction, training, and evaluation.

### Example Usage

Here is an example workflow to use the model:

1. **Positional Encoding**: The notebook first demonstrates the implementation of positional encoding, which helps the model understand the order of words in a sentence.

2. **Self-Attention Calculation**: The notebook shows how to compute scaled dot-product self-attention scores, which allows the model to focus on different parts of the input sentence.

3. **Masked Multi-Head Attention**: The notebook provides the implementation of masked multi-head attention for capturing different aspects of the input sequence concurrently.

4. **Build and Train the Transformer Model**: Finally, the notebook builds a full Transformer model and trains it on a sample dataset.

## Features

- **Positional Encoding**: Implements sinusoidal positional encodings to provide the model with information about the relative positions of words in a sentence.
- **Self-Attention Mechanism**: Efficiently computes self-attention scores to help the model focus on important parts of the input sequence.
- **Masked Multi-Head Attention**: Allows the model to attend to different parts of the input sequence simultaneously.
- **Transformer Model Construction**: Builds a full Transformer model using TensorFlow/Keras, incorporating all the elements of the Transformer architecture.
- **Model Training and Evaluation**: Trains the Transformer model on a sample dataset and evaluates its performance.

## Configuration

The notebook is designed to run with the default settings, but you may need to configure certain parameters based on your specific use case:

- **Model Hyperparameters**: You can adjust parameters such as the number of layers, attention heads, and embedding dimensions in the notebook cells where the Transformer model is constructed.
- **Training Parameters**: Modify the learning rate, batch size, and number of epochs in the training section to optimize the model's performance for your dataset.
- **Data Preprocessing**: Ensure that your input data is preprocessed correctly. The notebook uses tokenization and padding techniques from the Hugging Face `transformers` library. You can modify these methods based on your input data's format.
