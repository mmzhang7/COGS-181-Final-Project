import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import random
import time
from tqdm import tqdm
import json

from helpers import *
from model import *
from generate import *

def run_char_rnn(
    filename,
    model="gru",
    n_epochs=2000,
    print_every=100,
    hidden_size=100,
    n_layers=2,
    learning_rate=0.01,
    chunk_len=200,
    batch_size=100,
    shuffle=False,
    cuda=True,
    validation_split=0.2,  # Fraction of data to use for validation
):
    if cuda:
        print("Using CUDA")

    # Read and split the data
    file, file_len = read_file(filename)
    split_index = int(file_len * (1 - validation_split))
    train_file = file[:split_index]
    val_file = file[split_index:]
    train_file_len = len(train_file)
    val_file_len = len(val_file)

    def random_training_set(file, file_len, chunk_len, batch_size, shuffle):
        inp = torch.LongTensor(batch_size, chunk_len)
        target = torch.LongTensor(batch_size, chunk_len)
        for bi in range(batch_size):
            if shuffle:
                start_index = random.randint(0, file_len - chunk_len - 1)
            else:
                start_index = bi * (file_len // batch_size) % (file_len - chunk_len - 1)
            end_index = start_index + chunk_len + 1  # +1 to include the target character
            chunk = file[start_index:end_index]
            inp[bi] = char_tensor(chunk[:-1])  # Input is all characters except the last
            target[bi] = char_tensor(chunk[1:])  # Target is all characters except the first
        inp = inp
        target = target
        if cuda:
            inp = inp.cuda()
            target = target.cuda()
        return inp, target

    def evaluate(val_file, val_file_len, chunk_len, batch_size):
        decoder.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():  # Disable gradient calculation
            inp, target = random_training_set(val_file, val_file_len, chunk_len, batch_size, shuffle=False)
            hidden = decoder.init_hidden(batch_size)
            if cuda:
                if isinstance(hidden, tuple):
                    hidden = tuple(h.cuda() for h in hidden)
                else:
                    hidden = hidden.cuda()
            for c in range(chunk_len):
                output, hidden = decoder(inp[:, c], hidden)
                val_loss += criterion(output.view(batch_size, -1), target[:, c]).item()
        decoder.train()
        return val_loss / chunk_len

    def train(inp, target):
        hidden = decoder.init_hidden(batch_size)
        if cuda:
            if isinstance(hidden, tuple):
                hidden = tuple(h.cuda() for h in hidden) 
            else:
                hidden = hidden.cuda()  # For GRU or other cases where hidden is a single tensor
        decoder.zero_grad()
        loss = 0

        for c in range(chunk_len):
            output, hidden = decoder(inp[:, c], hidden)
            loss += criterion(output.view(batch_size, -1), target[:, c])

        loss.backward()
        decoder_optimizer.step()

        return loss.item() / chunk_len

    def save():
        save_filename = os.path.splitext(os.path.basename(filename))[0] + f"_{model}_h{hidden_size}_l{n_layers}_shuf{shuffle}_lr{learning_rate}_e{n_epochs}.pt"
        torch.save(decoder, save_filename)
        print('Saved as %s' % save_filename)

    # Initialize models and start training
    decoder = CharRNN(
        n_characters,
        hidden_size,
        n_characters,
        model=model,
        n_layers=n_layers,
    )
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if cuda:
        decoder.cuda()

    start = time.time()
    all_losses = []  # Store training losses
    val_losses = []  # Store validation losses
    best_val_loss = float('inf')  # Track the best validation loss

    try:
        print(f"Training {model} with hidden_size={hidden_size}, n_layers={n_layers}, lr={learning_rate}, shuffle={shuffle}, n_epochs={n_epochs}")
        for epoch in tqdm(range(1, n_epochs + 1)):
            # Train on the training set
            loss = train(*random_training_set(train_file, train_file_len, chunk_len, batch_size, shuffle))
            all_losses.append(loss)

            # Evaluate on the validation set
            if epoch % print_every == 0:
                val_loss = evaluate(val_file, val_file_len, chunk_len, batch_size)
                val_losses.append(val_loss)
                print(f'[Epoch {epoch}] Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')

                # Save the model if it has the best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save()

        print("Training completed. Saving final model...")
        save()

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        save()

    # Return the results for this combination
    return {
        "model_type": model,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "learning_rate": learning_rate,
        "shuffle": shuffle,
        "train_losses": all_losses,
        "val_losses": val_losses,
        "final_train_loss": all_losses[-1] if all_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
    }

# entropy.py
import torch
import torch.nn.functional as F

def compute_entropy(model, input_sequence, char_tensor, cuda=False):
    """
    Compute the entropy of the model's predictions for a given input sequence.
    
    Args:
        model: The trained CharRNN model.
        input_sequence (str): The input sequence of characters.
        char_tensor (function): A function to convert characters to tensors.
        cuda (bool): Whether to use GPU.
    
    Returns:
        list: A list of entropy values for each character in the sequence.
    """
    model.eval()  # Set the model to evaluation mode
    entropies = []
    hidden = model.init_hidden(1)  # Initialize hidden state for a single sequence

    for i in range(len(input_sequence) - 1):
        input_char = input_sequence[i]

        # Convert character to tensor
        input_tensor = char_tensor(input_char).unsqueeze(0)  # Add batch dimension
        if cuda:
            input_tensor = input_tensor.cuda()

        # Forward pass
        output, hidden = model(input_tensor, hidden)
        probs = F.softmax(output, dim=-1)

        # Compute entropy
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        entropies.append(entropy)

    return entropies