"""
# PCS5024 - Aprendizado Estatístico - Statistical Learning - 2025/1
### Professors:
### Anna Helena Reali Costa (anna.reali@usp.br)
### Fabio G. Cozman (fgcozman@usp.br)

Code Author: Marcel Barros (marcel.barros@usp.br)

Exercise: Teacher Forcing vs. Autoregressive Decoding in RNNs

Goal:
Using the Santos Port SSH dataset (or another time series), compare the
performance of the standard autoregressive encoder-decoder architecture
(as implemented in this script) with a modified version that incorporates
teacher forcing and curriculum learning during the decoding phase.

Background:
- Teacher Forcing (Deep Learning Book, 10.2.1): A training strategy where the
  model receives the ground-truth output from the prior time step as input,
  instead of its own prediction. This can stabilize training, especially early on.
  (URL: https://www.deeplearningbook.org/contents/rnn.html)
- Encoder-Decoder for Seq2Seq (Deep Learning Book, 10.4): The architecture used
  here maps an input sequence (past values) to an output sequence (future values),
  which may have different lengths.

The standard implementation in this script uses autoregressive decoding: the output
of one time step (`current_input` after linear projection) becomes the input for
the next step within the `decode` method.

PART 1: Teacher Forcing (5 points)

To implement teacher forcing, one could modify the `decode` loop.
This can be done in two ways:
1.  Feed Ground Truth: Instead of using the projected output (`current_input`)
    as the input to the RNN cell for the next step, use the corresponding
    ground-truth value from the target sequence `y`. This requires passing the
    target sequence `y` to the `decode` method during training.
2.  Combine Hidden States: Alternatively, project the ground-truth `y` value
    for the current step into the hidden space (e.g., using a separate linear
    layer) and combine it (e.g., add) with the hidden state `h_n` before or
    after the RNN cell computation.

You must run a experiment to compare the performance of the autoregressive
model with teacher forcing against the autoregressive model without
teacher forcing. Discuss the results.


PART 2: Curriculum Learning (5 points)

Curriculum learning is a training strategy where the model starts with
easier tasks and gradually progresses to harder ones. In this context,
it can be applied to teacher forcing by starting with a high probability
of using ground truth for the next step and gradually decreasing it.
This can help the model learn to rely on its own predictions over time.

You must implement a curriculum learning strategy
for teacher forcing and compare the performance of the three models:
1.  Autoregressive model (no teacher forcing)
2.  Teacher forcing model (with ground truth)
3.  Curriculum learning model (gradually decreasing teacher forcing)

Discuss the results.

Considerations:
- The Deep Learning Book (10.2.1) notes that teacher forcing can be applied even
  with hidden-to-hidden connections (like in GRU/LSTM), but BPTT is still needed
  if hidden states depend on previous steps.
- Teacher forcing is primarily a *training* technique. During inference/evaluation,
  the model typically needs to run autoregressively without ground truth.
  Your implementation should reflect this.


This script implements the *autoregressive* (non-teacher-forced) baseline
encoder-decoder model using a GRU. It encodes the past sequence and then
decodes the future sequence step-by-step, using the prediction from the
previous step as input for the current step. This setup is called an
autoregressive model.

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
import argparse
import uniplot

# --- Configuration ---
DEFAULT_PAST_LEN = 800
DEFAULT_FUTURE_LEN = 200
DEFAULT_SLIDING_WINDOW_STEP = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DATA_FILENAME = "santos_ssh.csv"
DEFAULT_TRAIN_TEST_SPLIT_DATE = "2020-06-01T00:00:00Z"
DEFAULT_PAST_PLOT_VIEW_SIZE = 200


def load_data(file_path: pathlib.Path) -> pl.DataFrame:
    """Loads data from CSV, and sets datetime and feature types.

    Args:
        file_path (pathlib.Path): Path to the CSV file.
    Returns:
        pl.DataFrame: Loaded and preprocessed DataFrame.
    """

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
    """Splits the data into training and testing sets based on the split date.

    Args:
        df (pl.DataFrame): DataFrame containing the data.
        split_date (datetime.datetime): Date to split the data.
    Returns:
        tuple: Training and testing DataFrames.
    """

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
    """Creates windows using a sliding window approach.

    Args:
        data (pl.DataFrame): DataFrame containing the data.
        past_len (int): Length of the past sequence.
        future_len (int): Length of the future sequence.
        step (int): Step size for the sliding window.

    Returns:
        tuple: Arrays of past and future sequences.
    """

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
    """Creates sequences and prepares PyTorch DataLoaders.

    Args:
        train_df (pl.DataFrame): Training DataFrame.
        test_df (pl.DataFrame): Testing DataFrame.
        past_len (int): Length of the past sequence.
        future_len (int): Length of the future sequence.
        batch_size (int): Batch size for DataLoader.
        sliding_window_step (int): Step size for sliding window.

    Returns:
        tuple: Training and testing DataLoaders.
    """

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

    # Add channel dimension
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
    """Autoregressive RNN Model using GRU."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        """Encodes the input sequence."""
        _, h_n = self.rnn(x)
        return h_n[0]  # h_n shape: (1, batch_size, hidden_size)

    def decode(self, h_n, target_seq_len):
        """Decodes the sequence autoregressively."""
        batch_size = h_n.shape[0]
        # Allocate memory for the output sequence
        output_seq = torch.empty(
            batch_size, target_seq_len, self.input_size, device=h_n.device
        )

        y_0 = self.linear(h_n)  # Project the last output of the encoder to input size

        current_input = y_0

        current_hidden = h_n.unsqueeze(0)  # Shape: (1, batch_size, hidden_size)

        for i in range(target_seq_len):
            # RNN expects input shape (batch, seq_len, features), seq_len is 1 here
            out, current_hidden = self.rnn(current_input.unsqueeze(1), current_hidden)
            # Project the output of the RNN step to the input size
            current_input = self.linear(out.squeeze(1))
            # Store the prediction
            output_seq[:, i] = current_input

        return output_seq

    def forward(self, x, target_seq_len):
        """Forward pass: encode the past, decode the future."""
        h_n = self.encode(x)
        output_seq = self.decode(h_n=h_n, target_seq_len=target_seq_len)
        return output_seq


# --- Training and Evaluation ---


def run_train_epoch(
    model: ARModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    model.train()

    progress_bar = tqdm(dataloader, desc="Training")

    losses = []
    # Use enumerate to get batch index for plotting
    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        target_seq_len = targets.shape[1]  # Get future length from target shape

        optimizer.zero_grad()

        outputs = model(inputs, target_seq_len)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().item())

    return np.mean(losses)


def run_eval_epoch(
    model: ARModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
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
            target_seq_len = targets.shape[1]  # Get future length from target shape
            predictions = model(inputs, target_seq_len)
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
    """Plots training and testing loss curves."""
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
    """Plots a single prediction example against the ground truth."""
    model.eval()
    with torch.no_grad():
        input_seq = X_test[idx].unsqueeze(0).to(device)  # Add batch dimension
        target_seq = y_test[idx].squeeze(-1).cpu().numpy()  # Remove channel dim

        prediction = (
            model(input_seq, future_len).squeeze(0).squeeze(-1).cpu().numpy()
        )  # Remove batch and channel dims

    input_seq_plot = (
        input_seq.squeeze(0).squeeze(-1).cpu().numpy()
    )  # Remove batch and channel dims

    plt.figure(figsize=(15, 6))
    # Plot past input
    plt.plot(range(past_len), input_seq_plot, label="Input (Past Data)", color="blue")
    # Plot ground truth future
    plt.plot(
        range(past_len, past_len + future_len),
        target_seq,
        label="Ground Truth (Future)",
        color="green",
        linestyle="--",
    )
    # Plot predicted future
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


def main(args):
    """Main function to run the training and evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Data Preparation ---
    file_path = pathlib.Path(args.data_filename)
    split_date = datetime.datetime.fromisoformat(args.split_date)
    df = load_data(file_path=file_path)
    feature_names = list(df.drop("datetime").columns)

    train_df, test_df = split_data(df=df, split_date=split_date)

    # --- Apply scaling ---
    # Calculate mean and std from training data only
    train_mean = train_df.select(feature_names).mean()
    train_std = train_df.select(feature_names).std()

    print(f"Scaling data using Train Mean: {train_mean}, Train Std: {train_std}")

    # Scale data
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
        past_len=int(args.past_len),
        future_len=int(args.future_len),
        batch_size=int(args.batch_size),
        sliding_window_step=int(args.sliding_window_step),
    )

    # --- Model Setup ---
    input_size = len(feature_names)
    model = ARModel(input_size=input_size, hidden_size=args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\n--- Starting Training ---")
    train_losses = []
    test_losses = []
    view_size = int(args.past_view_size)

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        # Training step
        train_loss = run_train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        train_losses.append(train_loss)
        print(f"Average Training Loss: {train_loss:.4f}")

        # Evaluation step
        test_loss, contexts, predictions, targets = run_eval_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device,
        )

        window_id = contexts.shape[0] // 2
        example_context = contexts[window_id, -view_size:]
        example_target = targets[window_id, :]
        example_prediction = predictions[window_id, :]

        example_context = example_context * train_std.item() + train_mean.item()
        example_target = example_target * train_std.item() + train_mean.item()
        example_prediction = example_prediction * train_std.item() + train_mean.item()

        # Plot only first feature
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
            height=15,  # Adjust height as needed
            lines=True,  # Use lines for time series
        )

        test_losses.append(test_loss)

        print(f"Average Test Loss: {test_loss:.4f}")
        # Plot overall loss curves with uniplot
        uniplot.plot(
            ys=[train_losses, test_losses],
            xs=[np.arange(1, epoch + 1)] * 2,  # Same x-axis for both
            color=True,
            legend_labels=["Train Loss", "Test Loss"],
            title=f"Epoch: {epoch} Loss Curves",
        )

    print("\n--- Training Complete ---")

    # --- Save Model ---
    model_save_path = pathlib.Path("model_weights.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- Results ---
    plot_results(train_losses, test_losses, args.num_epochs)

    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoregressive RNN Training Script")
    parser.add_argument(
        "--past_len",
        type=int,
        default=DEFAULT_PAST_LEN,
        help="Length of past sequence input.",
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=DEFAULT_FUTURE_LEN,
        help="Length of future sequence to predict.",
    )
    parser.add_argument(
        "--sliding_window_step",
        type=int,
        default=DEFAULT_SLIDING_WINDOW_STEP,
        help="Step size for sliding window.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Number of hidden units in RNN.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--data_filename",
        type=str,
        default=DEFAULT_DATA_FILENAME,
        help="Filename to save/load data.",
    )
    parser.add_argument(
        "--split_date",
        type=str,
        default=DEFAULT_TRAIN_TEST_SPLIT_DATE,
        help="Date string for train/test split.",
    )
    parser.add_argument(
        "--past_view_size",
        type=int,
        default=DEFAULT_PAST_PLOT_VIEW_SIZE,
        help="Number of past steps to show in uniplot.",
    )

    args = parser.parse_args()
    main(args)
