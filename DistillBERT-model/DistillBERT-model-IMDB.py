import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
import time
import tracemalloc
from codecarbon import EmissionsTracker


if __name__=='__main__':
    with EmissionsTracker() as tracker:
        st = time.time()
        tracemalloc.start()
        # Load data
        # Load data
        df = pd.read_csv('IMDB.csv')

        # Split data into train and test sets with a random seed of 42
        train_texts, test_texts, train_labels, test_labels = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

    # Load tokenizer and encode train and test texts with a max sequence length of 512
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', num_layers=6, output_hidden_states=True)
        train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
        test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512)

        # Convert labels to torch tensors
        train_labels = torch.tensor(train_labels.tolist())
        test_labels = torch.tensor(test_labels.tolist())

        # Load model
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, output_hidden_states=True)

        # Define optimizer and learning rate
        optimizer = AdamW(model.parameters(), lr=5e-5)

        # Define batch size, accumulation steps, and number of epochs
        batch_size = 64
        accumulation_steps = 4
        num_epochs = 3

        # Train model
        for epoch in range(num_epochs):
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']), torch.tensor(train_labels))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            batch_count = 0
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                loss = outputs.loss
                loss = loss / accumulation_steps
                loss.backward()
                batch_count += 1
                if batch_count % accumulation_steps == 0:
                    optimizer.step()

        # Evaluate model on test set
        test_dataset = torch.utils.data.TensorDataset(*[torch.tensor(test_encodings[e]) for e in ['input_ids', 'attention_mask']])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        correct = 0
        total = 0
        for batch in test_loader:
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, axis=1)
                labels = predictions.cpu().numpy()
                total += len(labels)
                correct += sum(labels == test_labels[total-len(labels):total].numpy())

        accuracy = correct / total
        print('Accuracy for IMDB with Multi-depth DistilBERT model:', accuracy)


        et = time.time()
        elapsed_time = et - st
        print('Execution time for Total:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for Total is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for IMDB with DistilBERT model:', tracker.final_emissions)
