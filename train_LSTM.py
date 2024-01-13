import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers):
        super(EncoderLSTM, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        
    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(src)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        # decoder_hidden: (batch_size, hidden_dim)
        
        # Calculate the attention scores.
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
        
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)

        return context_vector, attn_weights

class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers):
        super(DecoderLSTMWithAttention, self).__init__()
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.attention = Attention()

    def forward(self, input, encoder_outputs, hidden, cell):
        input = input.unsqueeze(1)  # (batch_size, 1)
        
        context_vector, attn_weights = self.attention(encoder_outputs, hidden[-1])  # using the last layer's hidden state

        rnn_input = torch.cat([input, context_vector.unsqueeze(1)], dim=2)  # (batch_size, 1, emb_dim + hidden_dim)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.out(output.squeeze(1))
        
        return prediction, hidden, cell

# Define your dataset and dataloaders (replace with your own dataset loading logic)
# For example, you might use torch.utils.data.TensorDataset and torch.utils.data.DataLoader

# train_dataset = ...
# test_dataset = ...
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
INPUT_DIM = 248  # Number of sensors
TIME_DIM = 160  # Number of time steps
OUTPUT_DIM = 4  # Number of classes for multi-class classification
EMB_DIM = 256 # Hyperparam?
HIDDEN_DIM = 512 # Hyperparam?
N_LAYERS = 2 # Hyperparam?

#Possible Hyperparams: learning rate, number of epochs, batch size, hidden dimension, embedding dimension, number of layers, what optimizer, what loss function

encoder = EncoderLSTM(INPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS)
decoder = DecoderLSTMWithAttention(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
def train_epoch(encoder, decoder, dataloader, criterion, optimizer):
    encoder.train()
    decoder.train()

    for batch in dataloader:
        input_data, target = batch
        optimizer.zero_grad()

        encoder_outputs, hidden, cell = encoder(input_data)
        output, _, _ = decoder(target, encoder_outputs, hidden, cell)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    return loss.item()

# Testing loop
def test(encoder, decoder, dataloader, criterion):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_data, target = batch

            encoder_outputs, hidden, cell = encoder(input_data)
            output, _, _ = decoder(target, encoder_outputs, hidden, cell)

            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)

    accuracy = correct_predictions / total_samples
    average_loss = total_loss / len(dataloader)

    return accuracy, average_loss

# Train the model for a few epochs
NUM_EPOCHS = 5

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(encoder, decoder, train_loader, criterion, optimizer)
    test_accuracy, test_loss = test(encoder, decoder, test_loader, criterion)

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS} | Training Loss: {train_loss:.4f} | Test Accuracy: {test_accuracy:.4f} | Test Loss: {test_loss:.4f}')
