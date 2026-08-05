import torch
import torch.nn as nn


def train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epochs
):
    """
    Train the Transformer model.
    """

    model.to(device)

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for source, target in train_loader:

            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(
                source,
                target[:, :-1]
            )

            loss = criterion(
                output.reshape(-1, output.shape[-1]),
                target[:, 1:].reshape(-1)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}/{epochs} Loss: {average_loss:.4f}"
        )

    return model


def save_model(
    model,
    path
):
    """
    Save trained model.
    """

    torch.save(
        model.state_dict(),
        path
    )


def load_model(
    model,
    path,
    device
):
    """
    Load trained model.
    """

    model.load_state_dict(
        torch.load(
            path,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model