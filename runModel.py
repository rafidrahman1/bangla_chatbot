import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model(model_path, tag2idx):
    # Load the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('sagorsarker/bangla-bert-base')
    model = BertForSequenceClassification.from_pretrained('sagorsarker/bangla-bert-base', num_labels=len(tag2idx))

    # Load the trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    return model, tokenizer

def get_reply(msg, model, tokenizer, tag2idx):
    # Preprocess the input
    inputs = tokenizer(msg, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    inputs['attention_mask'] = inputs['input_ids'].ne(tokenizer.pad_token_id).long()

    # Get the model's prediction
    with torch.no_grad():  # Disable gradient computation during inference
        outputs = model(**inputs)
        _, predicted = torch.max(outputs.logits, 1)

    # Convert the prediction to the corresponding tag
    idx2tag = {v: k for k, v in tag2idx.items()}
    return idx2tag[predicted.item()]

def main():
   
    tag2idx = {
        'Clever': 0, 'GoodBye': 1, 'TimeQuery': 2, 'Thanks': 3, 'CurrentHumanQuery': 4, 'NameQuery': 5, 'UnderstandQuery': 6, 'Pneumonia': 7, 'RealNameQuery': 8, 'Asking Query about Health': 9, 'NotTalking2U': 10, 'Jokes': 11, 'Result': 12, 'High Blood Pressure': 13, 'Dengue': 14, 'Tuberculosis': 15, 'Hepatitis B': 16, 'Normal Fever': 17, 'CourtesyGreeting': 18, 'Diabetes': 19, 'Allergy': 20, 'CourtesyGoodBye': 21, 'Greetings': 22, ' Low Blood Pressure': 23, 'Malaria': 24, 'Shutup': 25
    }
    model_path = 'E:\\Dev\\Github Repositories\\Bangla_chatbot\\Model\\model1.pth' 

    model, tokenizer = load_model(model_path, tag2idx)

    # Test the chatbot
    print(get_reply('এই যে', model, tokenizer, tag2idx))

if __name__ == "__main__":
    main()
