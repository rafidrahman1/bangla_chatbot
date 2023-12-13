Youtube video link of the presentation of the codebase: https://youtu.be/G4vH09Gpjgw

Documentation

Overview
Welcome to the documentation for the Bangla Chat Bot project! This document provides a comprehensive guide on how to use and understand the codebase for developing and running the Bangla Chat Bot with Neural Networks.

Introduction
The Bangla Chat Bot project leverages neural networks to enhance language understanding and generation capabilities in Bengali. The project includes components for Natural Language Understanding (NLU), contextual conversation, predefined responses, language support, and optional features like multimodal input and machine learning improvement.

Prerequisites
Before you begin, ensure you have the following prerequisites installed on your system:

- Python (>=3.6)
- PyTorch
- Transformers library
- Other dependencies (specified in `requirements.txt`)

Project Structure
The project is organized as follows:

- Main.py: Loads intents from a JSON file, converts tags to indices, defines and creates DataLoader, trains the BERT model, and saves it.

- ImportingTag.py: Loads intents from a JSON file, extracts tags, converts tags to indices, and prints the tag2idx dictionary.

- ImportEvadata.py: Loads intents from a JSON file, defines tag2idx, extracts patterns and tags for evaluation, and saves eval_tags to a text file.

- Evaluation.py: Loads the trained model and tokenizer, creates a DataLoader for evaluation, evaluates the model on test data, and prints evaluation loss and accuracy.

- RunModel.py: Loads the trained model and tokenizer, takes user input, and prints the chatbot's reply based on the input.

Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/rafidrahman1/Bangla_chatbot.git
   cd Bangla_chatbot
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Running the Chat Bot
Execute `RunModel.py` to run the chatbot. The script will prompt you to enter user input, and the chatbot will respond.

```bash
python RunModel.py
```

## Training the Model

To train the model with your own dataset, modify `Main.py` to load your data and adjust parameters as needed. Then, execute the script:

```bash
python Main.py
```

## Evaluation

Evaluate the model's performance on test data using `Evaluation.py`:

```bash
python Evaluation.py
```

Customization
Customize the project according to your requirements. Explore different pre-trained language models, adjust hyperparameters, or integrate additional features.

Contributing
Feel free to contribute to the project by submitting bug reports, feature requests, or pull requests. 
