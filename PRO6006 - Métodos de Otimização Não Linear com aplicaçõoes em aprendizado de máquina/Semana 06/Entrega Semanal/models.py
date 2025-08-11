import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, cell_type):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.cell_type = cell_type
        
        self.recurrent_layers = nn.ModuleList()
        
        layer_input_size = input_size
        for h_size in hidden_sizes:
            self.recurrent_layers.append(cell_type(layer_input_size, h_size))
            
            layer_input_size = h_size
    
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, input_seq):
        pass

    def sgd_step(self, parameters, learning_rate):
        with torch.no_grad():
            for param in parameters:
                if param.grad is not None:
                    param.data -= learning_rate * param.grad.data

    def zero_grad(self, parameters):
        for param in parameters:
            if param.grad is not None:
                param.grad.zero_()
                
    def train_loop(self, X_train, y_train, epochs=200, learning_rate=0.001):
        criterion = nn.MSELoss()
        
        print(f"\nStarting training for {self.__class__.__name__}...")
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            for i in range(len(X_train)):
                seq, target = X_train[i], y_train[i]
                
                self.zero_grad(self.parameters())
                y_pred = self(seq)
                loss = criterion(y_pred, target)
                loss.backward()
                self.sgd_step(self.parameters(), learning_rate)

                total_loss += loss.item()
                
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X_train):.6f}')
        
        print("Training complete.")
        return self

class RNN(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__(input_size, hidden_sizes, output_size, nn.RNNCell)
        
    def forward(self, input_seq):
        hidden_states = [torch.zeros(1, h_size) for h_size in self.hidden_sizes]

        for seq_input_t in input_seq:
            layer_input = seq_input_t.unsqueeze(0)
            next_hidden_states = []

            for i, rnn_layer in enumerate(self.recurrent_layers):
                h_prev_t = hidden_states[i]
                h_next_t = rnn_layer(layer_input, h_prev_t)
                layer_input = h_next_t
                next_hidden_states.append(h_next_t)
            
            hidden_states = next_hidden_states
            
        final_hidden_state = hidden_states[-1]
        output = self.output_layer(final_hidden_state)
        return output

class LSTM(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__(input_size, hidden_sizes, output_size, nn.LSTMCell)

    def forward(self, input_seq):
        h_states = [torch.zeros(1, h_size) for h_size in self.hidden_sizes]
        c_states = [torch.zeros(1, h_size) for h_size in self.hidden_sizes]

        for current_input in input_seq:
            layer_input = current_input.unsqueeze(0)
            next_h_states = []
            next_c_states = []

            for i, lstm_layer in enumerate(self.recurrent_layers):
                h_t, c_t = lstm_layer(layer_input, (h_states[i], c_states[i]))
                layer_input = h_t
                next_h_states.append(h_t)
                next_c_states.append(c_t)
            
            h_states = next_h_states
            c_states = next_c_states

        final_hidden_state = h_states[-1]
        output = self.output_layer(final_hidden_state)
        return output
