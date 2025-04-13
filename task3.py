
import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from TorchCRF import CRF
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to load and preprocess SQuAD v2 dataset
def load_squad_data(num_samples=15000):
    print("Loading SQuAD v2.0 dataset...")
    dataset = load_dataset("squad_v2")
    
    # Take a subset of the training data
    train_data = dataset["train"].shuffle(seed=42).select(range(num_samples))
    
    # Use the validation set as is
    val_data = dataset["validation"]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

# Custom dataset for question answering
class QuestionAnsweringDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384, stride=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        question = example["question"]
        context = example["context"]
        
        # For unanswerable questions, SQuAD v2 has empty answers
        is_impossible = len(example["answers"]["text"]) == 0
        
        if not is_impossible:
            answer_text = example["answers"]["text"][0]
            start_position_char = example["answers"]["answer_start"][0]
            
            # Find the end position character (start + length of answer)
            end_position_char = start_position_char + len(answer_text)
        else:
            # Set default values for unanswerable questions
            answer_text = ""
            start_position_char = 0
            end_position_char = 0
        
        # Tokenize the inputs
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            stride=self.stride,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        # Remove batch dimension added by the tokenizer
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Find token start and end positions from character positions
        if not is_impossible:
            # Get the offset mapping to convert char positions to token positions
            offset_mapping = encoding.pop("offset_mapping").tolist()
            
            # Find the start and end token positions
            start_position = 0
            end_position = 0
            
            for i, (start_char, end_char) in enumerate(offset_mapping):
                if start_char <= start_position_char < end_char:
                    start_position = i
                if start_char < end_position_char <= end_char:
                    end_position = i
                    break
        else:
            # For unanswerable questions, set positions to 0 (CLS token)
            start_position = 0
            end_position = 0
            encoding.pop("offset_mapping")
        
        # Add start and end positions to encoding
        encoding["start_positions"] = torch.tensor(start_position)
        encoding["end_positions"] = torch.tensor(end_position)
        encoding["is_impossible"] = torch.tensor(1 if is_impossible else 0)
        
        return encoding

# SpanBERT-CRF model
class SpanBERTCRF(nn.Module):
    def __init__(self, base_model_name="SpanBERT/spanbert-base-cased", num_labels=2):
        super(SpanBERTCRF, self).__init__()
        
        # Load base SpanBERT model
        self.spanbert = AutoModelForQuestionAnswering.from_pretrained(base_model_name)
        
        # Get hidden size from config
        hidden_size = self.spanbert.config.hidden_size
        
        # Start and end position classifiers
        self.start_classifier = nn.Linear(hidden_size, num_labels)
        self.end_classifier = nn.Linear(hidden_size, num_labels)
        
        # CRF layer without batch_first parameter
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        # Get the base model outputs
        outputs = self.spanbert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]
        
        # Get logits
        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(sequence_output)
        
        # If training
        loss = None
        if start_positions is not None and end_positions is not None:
            # Prepare inputs for CRF
            batch_size, seq_len = sequence_output.size(0), sequence_output.size(1)
            
            # Create masks for positions
            start_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)
            end_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)
            
            # Set the correct positions
            for i in range(batch_size):
                if start_positions[i] < seq_len:
                    start_mask[i, start_positions[i]] = 1
                if end_positions[i] < seq_len:
                    end_mask[i, end_positions[i]] = 1
            
            # Transpose tensors for CRF (seq_len, batch_size, ...)
            start_logits_t = start_logits.transpose(0, 1)
            end_logits_t = end_logits.transpose(0, 1)
            
            # Transpose masks as well
            start_mask_t = start_mask.transpose(0, 1)
            end_mask_t = end_mask.transpose(0, 1)
            attention_mask_t = attention_mask.transpose(0, 1)
            
            # CRF loss for start positions
            start_crf_loss = -self.crf(start_logits_t, start_mask_t, reduction='mean',
                                    mask=attention_mask_t.bool())
            
            # CRF loss for end positions
            end_crf_loss = -self.crf(end_logits_t, end_mask_t, reduction='mean',
                                    mask=attention_mask_t.bool())
            
            loss = start_crf_loss + end_crf_loss
        
        # If we're not training, decode using CRF
        else:
            # Transpose for CRF
            start_logits_t = start_logits.transpose(0, 1)
            end_logits_t = end_logits.transpose(0, 1)
            attention_mask_t = attention_mask.transpose(0, 1)
            
            # Decode start positions
            start_tags = self.crf.decode(start_logits_t, mask=attention_mask_t.bool())
            
            # Decode end positions
            end_tags = self.crf.decode(end_logits_t, mask=attention_mask_t.bool())
            
            # Convert tags to positions
            start_positions = []
            end_positions = []
            
            for batch_idx, (s_tags, e_tags) in enumerate(zip(start_tags, end_tags)):
                # Find first occurrence of tag 1 (if any)
                s_idx = next((i for i, t in enumerate(s_tags) if t == 1), 0)
                e_idx = next((i for i, t in enumerate(e_tags) if t == 1), 0)
                
                # If no tag 1 is found or end comes before start, set to 0
                if s_idx == 0 or e_idx == 0 or e_idx < s_idx:
                    s_idx, e_idx = 0, 0
                
                start_positions.append(torch.tensor(s_idx, device=device))
                end_positions.append(torch.tensor(e_idx, device=device))
            
            return torch.stack(start_positions), torch.stack(end_positions)
        
        return loss, start_logits, end_logits


# Training function for models
def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, model_name, tokenizer, is_crf=False):
    best_exact_match = 0
    train_losses = []
    val_exact_matches = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training {model_name}")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            if is_crf:
                loss, _, _ = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"]
                )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"]
                )
                loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        exact_match = evaluate_model(model, val_loader, model_name, tokenizer, is_crf)
        val_exact_matches.append(exact_match)
        
        print(f"Validation Exact Match: {exact_match:.2f}%")
        
        # Save the best model
        if exact_match > best_exact_match:
            best_exact_match = exact_match
            if is_crf:
                torch.save(model.state_dict(), f"{model_name}_best.pt")
            else:
                model.save_pretrained(f"{model_name}_best")
            print(f"Saved new best model with Exact Match: {exact_match:.2f}%")
    
    return train_losses, val_exact_matches

# Evaluation function
def exact_match_score(predictions, references):
    assert len(predictions) == len(references), "Lists must have the same length"
    matches = sum(p == r for p, r in zip(predictions, references))
    return matches / len(references) * 100  # Convert to percentage

def evaluate_model(model, val_loader, model_name, tokenizer, is_crf=False):
    model.eval()
    
    all_predictions = []
    all_references = []
    
    for batch in tqdm(val_loader, desc=f"Evaluating {model_name}"):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            if is_crf:
                start_positions, end_positions = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"]
                )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"]
                )
                start_positions = torch.argmax(outputs.start_logits, dim=1)
                end_positions = torch.argmax(outputs.end_logits, dim=1)
            
            # Extract answers from the tokens
            for i in range(len(start_positions)):
                start_idx = start_positions[i].item()
                end_idx = end_positions[i].item()
                
                # Handle unanswerable questions
                is_impossible = batch["is_impossible"][i].item()
                
                if is_impossible or start_idx == 0 or end_idx == 0 or end_idx < start_idx:
                    predicted_answer = ""
                else:
                    # Convert token indices to tokens to answer text
                    input_ids = batch["input_ids"][i].tolist()
                    answer_tokens = input_ids[start_idx:end_idx+1]
                    predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Get the reference answer
                if batch["is_impossible"][i].item():
                    reference_answer = ""
                else:
                    ref_start_idx = batch["start_positions"][i].item()
                    ref_end_idx = batch["end_positions"][i].item()
                    input_ids = batch["input_ids"][i].tolist()
                    ref_tokens = input_ids[ref_start_idx:ref_end_idx+1]
                    reference_answer = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                
                all_predictions.append(predicted_answer)
                all_references.append(reference_answer)
    
    # Calculate exact match score
    em_score = exact_match_score(all_predictions, all_references)
    
    return em_score

# Plot training results
def plot_results(spanbert_train_losses, spanbert_val_em, spanbert_crf_train_losses, spanbert_crf_val_em):
    plt.figure(figsize=(15, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(spanbert_train_losses, label='SpanBERT')
    plt.plot(spanbert_crf_train_losses, label='SpanBERT-CRF')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation exact match
    plt.subplot(1, 2, 2)
    plt.plot(spanbert_val_em, label='SpanBERT')
    plt.plot(spanbert_crf_val_em, label='SpanBERT-CRF')
    plt.title('Validation Exact Match Score')
    plt.xlabel('Epochs')
    plt.ylabel('Exact Match (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('spanbert_training_results.png')
    plt.show()

# Main function
def main():
    # Load dataset
    train_data, val_data = load_squad_data(num_samples=15000)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    
    # Create datasets
    train_dataset = QuestionAnsweringDataset(train_data, tokenizer)
    val_dataset = QuestionAnsweringDataset(val_data, tokenizer)
    
    # Create dataloaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training parameters
    num_epochs = 6
    learning_rate = 2e-5
    warmup_steps = 0
    total_steps = len(train_loader) * num_epochs
    
    # Fine-tune SpanBERT
    print("\nFine-tuning SpanBERT model...")
    spanbert_model = AutoModelForQuestionAnswering.from_pretrained("SpanBERT/spanbert-base-cased")
    spanbert_model.to(device)
    
    spanbert_optimizer = AdamW(spanbert_model.parameters(), lr=learning_rate)
    spanbert_scheduler = get_linear_schedule_with_warmup(
        spanbert_optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    spanbert_train_losses, spanbert_val_em = train_model(
        spanbert_model, 
        train_loader, 
        val_loader, 
        spanbert_optimizer, 
        spanbert_scheduler, 
        num_epochs, 
        "spanbert",
        tokenizer
    )
    
    # Fine-tune SpanBERT-CRF
    print("\nFine-tuning SpanBERT-CRF model...")
    spanbert_crf_model = SpanBERTCRF("SpanBERT/spanbert-base-cased")
    spanbert_crf_model.to(device)
    
    spanbert_crf_optimizer = AdamW(spanbert_crf_model.parameters(), lr=learning_rate)
    spanbert_crf_scheduler = get_linear_schedule_with_warmup(
        spanbert_crf_optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    spanbert_crf_train_losses, spanbert_crf_val_em = train_model(
        spanbert_crf_model, 
        train_loader, 
        val_loader, 
        spanbert_crf_optimizer, 
        spanbert_crf_scheduler, 
        num_epochs,
        "spanbert_crf",
        tokenizer,
        is_crf=True
    )
    
    # Plot and save results
    plot_results(spanbert_train_losses, spanbert_val_em, spanbert_crf_train_losses, spanbert_crf_val_em)
    
    # Final evaluation
    print("\nFinal Evaluation Results:")
    
    # Load best models
    best_spanbert = AutoModelForQuestionAnswering.from_pretrained("spanbert_best")
    best_spanbert.to(device)
    
    best_spanbert_crf = SpanBERTCRF("SpanBERT/spanbert-base-cased")
    best_spanbert_crf.load_state_dict(torch.load("spanbert_crf_best.pt"))
    best_spanbert_crf.to(device)
    
    # Evaluate
    spanbert_em = evaluate_model(best_spanbert, val_loader, "SpanBERT", tokenizer)
    spanbert_crf_em = evaluate_model(best_spanbert_crf, val_loader, "SpanBERT-CRF", tokenizer, is_crf=True)
    
    print(f"SpanBERT Exact Match: {spanbert_em:.2f}%")
    print(f"SpanBERT-CRF Exact Match: {spanbert_crf_em:.2f}%")
    
    # Save results to CSV
    results = {
        "Model": ["SpanBERT", "SpanBERT-CRF"],
        "Exact Match": [spanbert_em, spanbert_crf_em]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("spanbert_results.csv", index=False)
    print("Results saved to spanbert_results.csv")

if __name__ == "__main__":
    main()
'''



import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Function to load the fine-tuned model
def load_qa_model(model_path="spanbert_best"):
    """
    Load the fine-tuned SpanBERT model from Task 3
    """
    print(f"Loading model from {model_path}...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

# Function to predict answers
def predict_answer(model, tokenizer, device, question, context):
    """
    Extract answer from context based on question using the fine-tuned model
    """
    # Tokenize input
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        stride=128,
        padding='max_length',
        truncation='only_second',
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
    
    # Get the most likely answer span
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()
    
    # If we have an impossible answer or invalid span
    if start_idx == 0 or end_idx == 0 or end_idx < start_idx:
        return "No answer found in the given context."
    
    # Convert token positions to tokens to answer text
    input_ids = inputs['input_ids'][0].tolist()
    answer_tokens = input_ids[start_idx:end_idx+1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Clean up the answer (optional refinement)
    answer = answer.strip()
    
    return answer

# Example usage
def main():
    # Load model
    model, tokenizer, device = load_qa_model()
    
    # Example from the task description
    context = "Beyoncé Giselle Knowles-Carter (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'."
    
    question = "When did Beyonce start becoming popular?"
    
    # Get answer
    answer = predict_answer(model, tokenizer, device, question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Additional examples to test
    examples = [
        {
            "question": "What was the name of Beyonce's first solo album?",
            "context": context
        },
        {
            "question": "Who managed Destiny's Child?",
            "context": context
        },
        {
            "question": "How many Grammy Awards did her debut album earn?",
            "context": context
        }
    ]
    
    for example in examples:
        answer = predict_answer(model, tokenizer, device, example["question"], example["context"])
        print(f"\nQuestion: {example['question']}")
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

'''