# Simple training script for Arabic teacher notes classification
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
# Load data
df = pd.read_csv("data/teacher_notes_labeled.csv")
print(f"Loaded {len(df)} samples")
print("Label distribution:")
print(df['label'].value_counts())

# Map labels to numbers
label_map = {"low": 0, "moderate": 1, "high": 2}
df["label_num"] = df["label"].map(label_map)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["teacher_note"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)

print(f"\nTraining samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Load tokenizer and model
print("\nLoading Arabic BERT model...")
tokenizer = BertTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
model = BertForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv2", num_labels=3)

# Simple training loop
print("\nStarting training...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Tokenize training data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
train_labels_tensor = torch.tensor(list(train_labels))

# Simple training with small batches
batch_size = 4
num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0
    num_batches = len(train_texts) // batch_size + (1 if len(train_texts) % batch_size != 0 else 0)

    for i in range(0, len(train_texts), batch_size):
        # Get batch
        batch_texts = list(train_texts)[i:i+batch_size]
        batch_labels = list(train_labels)[i:i+batch_size]

        # Tokenize batch
        batch_encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        batch_labels_tensor = torch.tensor(batch_labels)

        # Forward pass
        outputs = model(**batch_encodings, labels=batch_labels_tensor)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress
        batch_num = i // batch_size + 1
        if batch_num % 2 == 0 or batch_num == num_batches:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_num}/{num_batches}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

print("\nTraining completed!")

# Evaluate on validation set
print("Evaluating on validation set...")
model.eval()
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128, return_tensors="pt")

with torch.no_grad():
    outputs = model(**val_encodings)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Calculate accuracy
accuracy = accuracy_score(val_labels, predictions.numpy())
print(f"Validation Accuracy: {accuracy:.3f}")

# Print classification report
target_names = ["low", "moderate", "high"]
print("\nClassification Report:")
print(classification_report(val_labels, predictions.numpy(), target_names=target_names))

# Save model
print("\nSaving model...")
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
print("Model saved successfully!")

# Test with sample predictions
print("\nTesting with sample notes:")
test_notes = [
    "الطالب غاب 3 مرات ولم يسلم الواجب",
    "الطالبة مشاركة ومتفوقة",
    "الطالب يحتاج مساعدة إضافية في الرياضيات"
]

for note in test_notes:
    inputs = tokenizer(note, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
        label = target_names[prediction.item()]
        confidence = torch.softmax(outputs.logits, dim=-1).max().item()
        print(f"Note: {note}")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
        print()
