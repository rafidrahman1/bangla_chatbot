import json

# Load intents from JSON file
with open('BanglaHealthcareChatbotData.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define your tag2idx
tag2idx = {
    'Clever': 0, 'GoodBye': 1, 'TimeQuery': 2, 'Thanks': 3, 'CurrentHumanQuery': 4, 'NameQuery': 5, 'UnderstandQuery': 6, 'Pneumonia': 7, 'RealNameQuery': 8, 'Asking Query about Health': 9, 'NotTalking2U': 10, 'Jokes': 11, 'Result': 12, 'High Blood Pressure': 13, 'Dengue': 14, 'Tuberculosis': 15, 'Hepatitis B': 16, 'Normal Fever': 17, 'CourtesyGreeting': 18, 'Diabetes': 19, 'Allergy': 20, 'CourtesyGoodBye': 21, 'Greetings': 22, ' Low Blood Pressure': 23, 'Malaria': 24, 'Shutup': 25
}

# Extract patterns and tags from intents
eval_patterns = []
eval_tags = []
for intent in data['intents']:
    if intent['tag'] in tag2idx:  # Only include patterns and tags that are in tag2idx
        for pattern in intent['patterns']:
            eval_patterns.append(pattern)
            eval_tags.append(tag2idx[intent['tag']])

# Save eval_patterns and eval_tags to a text file
with open('eval_tag.txt', 'w', encoding='utf-8') as file:
    file.write(', '.join([f"{tag}" for tag in eval_tags]))

print('Data saved to eval_data.txt.')
