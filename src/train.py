import torch
import torch.optim as optim
from typing import List, Dict

try:
    from src.skipgram import SkipGramNeg, NegativeSamplingLoss  
    from src.data_processing import get_batches, cosine_similarity
except ImportError:
    from skipgram import SkipGramNeg, NegativeSamplingLoss
    from data_processing import get_batches, cosine_similarity

def train_skipgram(model: SkipGramNeg,
                   words: List[int], 
                   int_to_vocab: Dict[int, str], 
                   batch_size=512, 
                   epochs=5, 
                   learning_rate=0.003, 
                   window_size=5, 
                   print_every=1500,
                   device='cpu'):
    """Trains the SkipGram model using negative sampling.

    Args:
        model: The SkipGram model to be trained.
        words: A list of words (integers) to train on.
        int_to_vocab: A dictionary mapping integers back to vocabulary words.
        batch_size: The size of each batch of input and target words.
        epochs: The number of epochs to train for.
        learning_rate: The learning rate for the optimizer.
        window_size: The size of the context window for generating training pairs.
        print_every: The frequency of printing the training loss and validation examples.
        device: The device (CPU or GPU) where the tensors will be allocated.
    """
    # Define loss and optimizer
    # TODO
    criterion = None
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    # Training loop
    for epoch in range(epochs):
        for input_words, target_words in None:
            steps += 1
            # Convert inputs and context words into tensors
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
            inputs, targets = inputs.to(device), targets.to(device)

            # input, output, and noise vectors
            # TODO
            input_vectors = None
            output_vectors = None
            noise_vectors = None
            
            # negative sampling loss
            # TODO
            loss = criterion(None, None, None)

            # Backward step
            # TODO

            if steps % print_every == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Step: {steps}, Loss: {loss.item()}")
                # Cosine similarity
                # TODO
                valid_examples, valid_similarities = cosine_similarity(None, device=device)
                _, closest_idxs = valid_similarities.topk(6)

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...\n")
