import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Loading intents from JSON file
with open('BanglaHealthcareChatbotData.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract patterns and tags from intents
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Convert tags to indices
tag2idx = {'Clever': 0, 'High Blood Pressure': 1, 'NotTalking2U': 2, 'Normal Fever': 3, 'Thanks': 4, 'Tuberculosis': 5, 'Pneumonia': 6, 'Shutup': 7, 'Result': 8, 'CurrentHumanQuery': 9, 'CourtesyGreeting': 10, 'Malaria': 11, 'UnderstandQuery': 12, 'GoodBye': 13, 'Diabetes': 14, 'TimeQuery': 15, 'NameQuery': 16, 'Jokes': 17, 'RealNameQuery': 18, 'Asking Query about Health': 19, 'Dengue': 20, ' Low Blood Pressure': 21, 'Allergy': 22, 'CourtesyGoodBye': 23, 'Greetings': 24, 'Hepatitis B': 25}
tags = [tag2idx[tag] for tag in tags]

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('sagorsarker/bangla-bert-base')
model = BertForSequenceClassification.from_pretrained('sagorsarker/bangla-bert-base', num_labels=len(tag2idx))

# Defining dataset
class ChatDataset(Dataset):
    def __init__(self, patterns, tags):
        self.patterns = patterns
        self.tags = tags

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        tag = self.tags[idx]
        inputs = tokenizer(pattern, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        inputs['labels'] = torch.tensor([tag])
        inputs['attention_mask'] = inputs['input_ids'].ne(tokenizer.pad_token_id).long()  # Create attention_mask
        return inputs

# Creating DataLoader
dataset = ChatDataset(patterns, tags)
dataloader = DataLoader(dataset, batch_size=32)

# Training model
optimizer = optim.Adam(model.parameters())
for epoch in range(10):  # Number of epochs
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].squeeze()  # Removing extra dimension from input_ids
        attention_mask = batch['attention_mask'].squeeze()  # Removing extra dimension from attention_mask
        labels = batch['labels'].squeeze()  # Removing extra dimension from labels
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Ensuring directory exists
model_dir = os.path.dirname('E:\\Dev\\Github Repositories\\Bangla_chatbot\\Model\\model1.pth')
os.makedirs(model_dir, exist_ok=True)

# Saving model
torch.save(model.state_dict(), 'E:\\Dev\\Github Repositories\\Bangla_chatbot\\Model\\model1.pth')