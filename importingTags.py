import json

# Load intents from JSON file
with open('BanglaHealthcareChatbotData.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract tags from intents
tags = []
for intent in data['intents']:
    tags.append(intent['tag'])

# Convert tags to indices
tag2idx = {tag: idx for idx, tag in enumerate(set(tags))}

# Print tag2idx
print(tag2idx)
