import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os

# --- Configuration ---
MODEL_NAME = 'gpt2'
OUTPUT_DIR = './results'
TRAIN_FILE = 'train_data.txt' # Placeholder for your training data file
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Adjust based on your GPU memory
SAVE_STEPS = 10_000
SAVE_TOTAL_LIMIT = 2

def create_sample_train_file():
    """Creates a dummy training file if it doesn't exist."""
    if not os.path.exists(TRAIN_FILE):
        print(f"'{TRAIN_FILE}' not found. Creating a sample file.")
        with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
            f.write("Hello, this is a sample sentence for training.\n")
            f.write("GPT-2 is a powerful language model.\n")
            f.write("Fine-tuning helps adapt it to specific tasks.\n")
            f.write("This is another example line of text.\n")
            f.write("The model will learn from these examples.\n")
        print(f"Sample '{TRAIN_FILE}' created. Replace it with your actual training data.")

def load_dataset(file_path, tokenizer, block_size=128):
    """Loads and tokenizes the dataset."""
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def main():
    """Main function to fine-tune and generate text."""
    # Create a sample training file if none exists
    create_sample_train_file()

    # --- 1. Load Tokenizer and Model ---
    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Add a padding token if it doesn't exist (GPT-2 usually doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # --- 2. Load and Prepare Dataset ---
    print(f"Loading and preparing dataset from '{TRAIN_FILE}'...")
    train_dataset = load_dataset(TRAIN_FILE, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False # MLM is for Masked Language Modeling, not needed for GPT-2
    )

    # --- 3. Set up Training Arguments ---
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_dir='./logs', # Directory for storing logs
        logging_steps=100,
        # evaluation_strategy="steps", # Uncomment if you have an eval dataset
        # eval_steps=500, # Uncomment if you have an eval dataset
    )

    # --- 4. Initialize Trainer and Start Training ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # Uncomment if you have an eval dataset
    )

    print("Starting fine-tuning...")
    trainer.train()

    # --- 5. Save the Fine-tuned Model ---
    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved successfully.")

    # --- 6. Generate Text with the Fine-tuned Model ---
    print("\n--- Generating text with the fine-tuned model ---")
    prompt = "Once upon a time" # Example prompt
    print(f"Prompt: {prompt}")

    # Load the fine-tuned model and tokenizer
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR)
    fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)

    inputs = fine_tuned_tokenizer.encode(prompt, return_tensors='pt')
    
    # Ensure the model is on the correct device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tuned_model.to(device)
    inputs = inputs.to(device)

    # Generate text
    # Adjust max_length, num_return_sequences, temperature, etc. as needed
    outputs = fine_tuned_model.generate(
        inputs, 
        max_length=100, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, # Prevents repeating n-grams
        early_stopping=True, # Stop generation when EOS token is produced
        temperature=0.7, # Controls randomness: lower is more deterministic
        top_k=50, # Considers the top K most likely next words
        top_p=0.95 # Nucleus sampling: considers the smallest set of words whose cumulative probability exceeds P
    )

    generated_text = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text:\n{generated_text}")

if __name__ == "__main__":
    main()