import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_hidden = 8

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_hidden),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def show_images(images, labels):
    """
    Display a set of images and their labels using matplotlib.
    The first column of `images` should contain the image indices,
    and the second column should contain the flattened image pixels
    reshaped into 28x28 arrays.
    """
    # Extract the image indices and reshaped pixels
    pixels = images.reshape(-1, 28, 28)

    # Create a figure with subplots for each image
    fig, axs = plt.subplots(
        ncols=len(images), nrows=1, figsize=(10, 3 * len(images))
    )

    # Loop over the images and display them with their labels
    for i in range(len(images)):
        # Display the image and its label
        axs[i].imshow(pixels[i], cmap="gray")
        axs[i].set_title("Label: {}".format(labels[i]))

        # Remove the tick marks and axis labels
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel("Index: {}".format(i))

    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.5)

    # Show the figure
    plt.show()


def one_hot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


if __name__ == "__main__":
    from dataset.dataset import load_mnist

    X_train, X_test, y_train, y_test = load_mnist()
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    X_train /= 255.0
    X_test /= 255.0

    y_train = one_hot(y_train.flatten())
    y_test = one_hot(y_test.flatten())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10

    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.to(device)

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            encoded, decoded = model(data)
            loss = criterion(decoded, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(
            "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
        )
