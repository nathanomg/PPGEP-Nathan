"""
# PCS5024 - Aprendizado Estatístico - Statistical Learning - 2025/1
### Professors:
### Anna Helena Reali Costa (anna.reali@usp.br)
### Fabio G. Cozman (fgcozman@usp.br)

Code Author: Marcel Barros (marcel.barros@usp.br)
Exercise: Teacher Forcing vs. Autoregressive Decoding in RNNs

##################

Student Name: Nathan Sampaio Santos
NUSP: 8988661

##################

Exercise Comments

PART 1: Teacher Forcing

O Teacher Forcing foi implementado de duas maneiras, conforme sugestão do enunciado.
Na primeira delas, o método 'decode' da classe ARModel recebe um parâmetro booleano 'teacher_forcing'. 
Caso este parâmetro seja verdadeiro, o modelo utiliza os valores reais do alvo (ground truth) como entrada para a próxima etapa de previsão. 
Caso contrário, ele utiliza a previsão anterior como entrada. Esta abordagem é útil para treinar o modelo com dados reais, 
e o seu resultado pode ser visto na imagem 'loss_curve [teacher forcing]', 
onde o tempo de convergência do conjunto de teste foi considerávelmente menor do que o original.

O segundo método implementado foi o 'combine_hidden_states', que combina o estado oculto do modelo com a projeção do alvo (ground truth) no espaço oculto. 
Isso é feito para permitir que o modelo aprenda a ajustar suas previsões com base nas informações reais do alvo, melhorando a precisão das previsões futuras. 
Além disso, o modelo possui uma estabilidade melhorada em relação ao método anterior, pois não depende exclusivamente das previsões anteriores.
O resultado desta abordagem pode ser visto na imagem 'loss_curve [combine hidden states]'.
Apesar da convergência se estabilizar tão rapidamente quanto o método anterior, a curva constante de perda do conjunto de teste indica que algo não foi corretamente implementado, 
pois o modelo não está aprendendo a generalizar os dados de teste.
Assim, esse teste serve para aprimorar a discussão do desenvolvimento do modelo, porém não é uma abordagem que está pronta para ser utilizada do modo em que foi implementada. 


PART 2: Curriculum Learning

Uma outra abordagem implementada foi o Curriculum Learning, que consiste em treinar o modelo com uma taxa de Teacher Forcing que diminui gradualmente ao longo das épocas.
Neste caso, o modelo começa treinando com 100% de Teacher Forcing e, ao longo das épocas, essa taxa diminui até chegar a 0%. 
Em outras palavras, o modelo começa aprendendo com os dados reais e, gradualmente, passa a depender mais de suas próprias previsões.
A vantagem dessa abordagem é que ela permite que o modelo aprenda a generalizar melhor,
reduzindo o risco de overfitting e melhorando a capacidade de previsão em dados não vistos. 
O resultado desta abordagem pode ser visto na imagem 'loss_curve [curriculum learning]', onde os resultados de teste convergem para valores mais baixos ao longo das épocas,
o que indica uma melhoria na capacidade de generalização do modelo. Entretanto, o resultado do conjunto de treinamento passou a crescer após a metade das épocas,
o que indica que o modelo pode ter começado a overfitar os dados de treinamento.
Apesar de tudo, o erro para o conjunto de teste foi o menor entre todos os métodos testados, o que indica que o modelo está aprendendo a generalizar os dados de teste de maneira mais eficaz.

"""


import datetime
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import uniplot

# --- Configuration ---
DEFAULT_PAST_LEN = 800
DEFAULT_FUTURE_LEN = 100
DEFAULT_SLIDING_WINDOW_STEP = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_EPOCHS = 20
DEFAULT_DATA_FILENAME = "santos_ssh.csv"
DEFAULT_TRAIN_TEST_SPLIT_DATE = "2020-06-01T00:00:00Z"
DEFAULT_PAST_PLOT_VIEW_SIZE = 200


def load_data(file_path: pathlib.Path) -> pl.DataFrame:
    df = pl.read_csv(file_path)
    df = df.with_columns(
        [
            pl.col("datetime").str.to_datetime(time_unit="ms", strict=True, exact=True),
        ]
        + [pl.col(f).cast(pl.Float32) for f in df.columns if f != "datetime"]
    )
    return df


def split_data(
    df: pl.DataFrame, split_date: datetime.datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_df = df.filter(pl.col("datetime") < split_date)
    test_df = df.filter(pl.col("datetime") >= split_date)

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    return train_df, test_df


def create_sequences(
    df: pl.DataFrame,
    past_len: int,
    future_len: int,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(past_len, len(df) - future_len, step):
        x = df[(i - past_len) : i]
        y = df[i : (i + future_len)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def prepare_dataloaders(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    past_len: int,
    future_len: int,
    batch_size: int,
    sliding_window_step: int,
):
    print("Creating sequences and dataloaders...")
    x_train, y_train = create_sequences(
        df=train_df,
        past_len=past_len,
        future_len=future_len,
        step=sliding_window_step,
    )
    x_test, y_test = create_sequences(
        df=test_df,
        past_len=past_len,
        future_len=future_len,
        step=sliding_window_step,
    )

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_dataloader,
        test_dataloader,
    )


# --- Model Definition ---


class ARModel(nn.Module):
    """Autoregressive RNN Model using GRU with teacher forcing and hidden state combination."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

        # Linear layer to project ground truth input into hidden space
        self.gt_to_hidden = nn.Linear(input_size, hidden_size)

    def encode(self, x):
        _, h_n = self.rnn(x)
        return h_n[0]

    def decode(self, h_n, target_seq_len, target_seq=None, teacher_forcing=False, combine_hidden_states=False):
        batch_size = h_n.shape[0]
        device = h_n.device

        output_seq = torch.empty(batch_size, target_seq_len, self.input_size, device=device)

        current_input = self.linear(h_n)
        current_hidden = h_n.unsqueeze(0)

        for i in range(target_seq_len):
            # If teacher forcing is enabled, use ground truth
            if teacher_forcing and target_seq is not None and i < target_seq_len - 1:
                current_input = target_seq[:, i, :]

            # Optionally combine the ground truth projection with the hidden state
            if combine_hidden_states and target_seq is not None and i < target_seq_len - 1:
                gt_proj = self.gt_to_hidden(target_seq[:, i, :])  # Project ground truth to hidden space
                combined_hidden = current_hidden.squeeze(0) + gt_proj  # Combine with hidden state
                current_hidden = combined_hidden.unsqueeze(0)  # Restore RNN shape

            out, current_hidden = self.rnn(current_input.unsqueeze(1), current_hidden)
            current_input = self.linear(out.squeeze(1))
            output_seq[:, i] = current_input

        return output_seq

    def forward(self, x, target_seq_len, target_seq=None, teacher_forcing=False, combine_hidden_states=False):
        h_n = self.encode(x)
        output_seq = self.decode(h_n, target_seq_len, target_seq, teacher_forcing, combine_hidden_states)
        return output_seq


def run_train_epoch(
    model: ARModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float = 1.0,  # Default to 100% teacher forcing
    combine_hidden_states: bool = False,
    curriculum_learning: bool = False,  # If curriculum learning is enabled
):
    model.train()
    progress_bar = tqdm(dataloader, desc="Training")
    losses = []

    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        target_seq_len = targets.shape[1]

        optimizer.zero_grad()

        # If curriculum learning is enabled, decrease teacher forcing ratio over epochs
        teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

        outputs = model(
            inputs,
            target_seq_len,
            target_seq=targets,
            teacher_forcing=teacher_forcing,
            combine_hidden_states=combine_hidden_states,
        )
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def run_eval_epoch(
    model: ARModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    combine_hidden_states: bool = False,
):
    model.eval()
    progress_bar = tqdm(dataloader, desc="Testing")

    total_loss = 0.0
    num_batches = 0
    all_contexts = []
    all_targets = []
    all_predictions = []

    for inputs, targets in progress_bar:
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            target_seq_len = targets.shape[1]

            predictions = model(
                inputs,
                target_seq_len,
                teacher_forcing=False,  # No teacher forcing during evaluation
                combine_hidden_states=False, # No teacher forcing during evaluation
            )

            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())

            all_contexts.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    avg_loss = total_loss / num_batches
    return (
        avg_loss,
        np.concatenate(all_contexts, axis=0),
        np.concatenate(all_predictions, axis=0),
        np.concatenate(all_targets, axis=0),
    )


def plot_results(train_losses, test_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epoch + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss (MSE)")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()
    plt.close()


def plot_prediction(model, X_test, y_test, device, past_len, future_len, idx=0):
    model.eval()
    with torch.no_grad():
        input_seq = X_test[idx].unsqueeze(0).to(device)
        target_seq = y_test[idx].squeeze(-1).cpu().numpy()

        prediction = (
            model(input_seq, future_len).squeeze(0).squeeze(-1).cpu().numpy()
        )

    input_seq_plot = input_seq.squeeze(0).squeeze(-1).cpu().numpy()

    plt.figure(figsize=(15, 6))
    plt.plot(range(past_len), input_seq_plot, label="Input (Past Data)", color="blue")
    plt.plot(
        range(past_len, past_len + future_len),
        target_seq,
        label="Ground Truth (Future)",
        color="green",
        linestyle="--",
    )
    plt.plot(
        range(past_len, past_len + future_len),
        prediction,
        label="Prediction (Future)",
        color="red",
        linestyle="-.",
    )

    plt.xlabel("Time Steps")
    plt.ylabel("SSH Value")
    plt.title(f"Example Prediction vs. Ground Truth (Index {idx})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"prediction_example_{idx}.png")
    plt.show()
    plt.close()


def run_model(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract parameters from the dictionary
    file_path = pathlib.Path(params['data_filename'])
    split_date = datetime.datetime.fromisoformat(params['split_date'])
    df = load_data(file_path=file_path)
    feature_names = list(df.drop("datetime").columns)

    train_df, test_df = split_data(df=df, split_date=split_date)

    train_mean = train_df.select(feature_names).mean()
    train_std = train_df.select(feature_names).std()

    print(f"Scaling data using Train Mean: {train_mean}, Train Std: {train_std}")

    train_data_scaled = train_df.with_columns(
        [
            (pl.col(f) - train_mean.select([f]).item()) / train_std.select([f]).item()
            for f in feature_names
        ]
    )
    test_data_scaled = test_df.with_columns(
        [
            (pl.col(f) - train_mean.select([f]).item()) / train_std.select([f]).item()
            for f in feature_names
        ]
    )

    train_dataloader, test_dataloader = prepare_dataloaders(
        train_df=train_data_scaled.select(feature_names),
        test_df=test_data_scaled.select(feature_names),
        past_len=params['past_len'],
        future_len=params['future_len'],
        batch_size=params['batch_size'],
        sliding_window_step=params['sliding_window_step'],
    )

    input_size = len(feature_names)
    model = ARModel(input_size=input_size, hidden_size=params['hidden_size']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    print("\n--- Starting Training ---")
    train_losses = []
    test_losses = []
    view_size = params['past_view_size']

    for epoch in range(1, params['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{params['num_epochs']}")

        # Gradually decrease the teacher forcing ratio (curriculum learning)
        teacher_forcing_ratio = max(0.0, 1.0 - (epoch / params['num_epochs'])) if params['teacher_forcing'] else 0

        train_loss = run_train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=teacher_forcing_ratio,  # Pass the ratio
            combine_hidden_states=params['combine_hidden_states'],  
            curriculum_learning=params['curriculum_learning'],  
        )
        train_losses.append(train_loss)
        print(f"Average Training Loss: {train_loss:.4f}")

        test_loss, contexts, predictions, targets = run_eval_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device,
            combine_hidden_states=params['combine_hidden_states'],
        )

        window_id = contexts.shape[0] // 2
        example_context = contexts[window_id, -view_size:]
        example_target = targets[window_id, :]
        example_prediction = predictions[window_id, :]

        example_context = example_context * train_std.item() + train_mean.item()
        example_target = example_target * train_std.item() + train_mean.item()
        example_prediction = example_prediction * train_std.item() + train_mean.item()

        example_context = example_context[:, 0]
        example_target = example_target[:, 0]
        example_prediction = example_prediction[:, 0]

        uniplot.plot(
            ys=[
                example_target,
                example_context,
                example_prediction,
            ],
            xs=[
                np.arange(0, example_target.shape[0]),
                np.arange(-view_size, 0),
                np.arange(0, example_prediction.shape[0]),
            ],
            color=True,
            legend_labels=["Target", "Context", "Prediction"],
            title=f"Epoch: {epoch}, Eval Element: {window_id}, Loss: {test_loss:.4f}",
            height=15,
            lines=True,
        )

        test_losses.append(test_loss)

        print(f"Average Test Loss: {test_loss:.4f}")

        uniplot.plot(
            ys=[train_losses, test_losses],
            xs=[np.arange(1, epoch + 1)] * 2,
            color=True,
            legend_labels=["Train Loss", "Test Loss"],
            title=f"Epoch: {epoch} Loss Curves",
        )

    print("\n--- Training Complete ---")

    model_save_path = pathlib.Path("model_weights.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plot_results(train_losses, test_losses, params['num_epochs'])

    print("Script finished.")


if __name__ == "__main__":

    params = {
        "data_filename": "santos_ssh.csv",
        "split_date": "2020-06-01T00:00:00Z",
        "past_len": DEFAULT_PAST_LEN,
        "future_len": DEFAULT_FUTURE_LEN,
        "batch_size": DEFAULT_BATCH_SIZE,
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "num_epochs": DEFAULT_NUM_EPOCHS,
        "learning_rate": 1e-4,
        "teacher_forcing": True,  # Enable teacher forcing
        "combine_hidden_states": True,  # Combine hidden states
        "curriculum_learning": False,  # Enable curriculum learning
        "past_view_size": DEFAULT_PAST_PLOT_VIEW_SIZE,
        "sliding_window_step": DEFAULT_SLIDING_WINDOW_STEP,
    }
    
    run_model(params)

