import numpy as np
import torch.nn as nn
import torch


class AutoEncoderNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        """
        Your autoencoder model definition should go here
        """
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function should map a given data matrix onto the bottleneck hidden layer
        :param x: the input data matrix of type torch.Tensor
        :return: the resulting projected data matrix of type torch.Tensor
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Your autoencoder model's forward pass operations should go here
        :param x: the input data matrix of type torch array
        :return: the neural network output as torch array
        """
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x


class AutoEncoder:

    def __init__(self, input_dim: int, projection_dim: int, learning_rate: float, iteration_count: int):
        """
        Initializes the Auto Encoder method
        :param input_dim: the input data space dimensionality
        :param projection_dim: the projection space dimensionality
        :param learning_rate: the learning rate for the auto encoder neural network training
        :param iteration_count: the number epoch for the neural network training
        """
        self.input_dim = input_dim
        # self.projection_matrix = projection_dim  # What is this
        self.projection_dim = projection_dim
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        # self.autoencoder_model = AutoEncoder(input_dim, projection_dim) # that is wrong i guess?
        self.autoencoder_model = AutoEncoderNetwork(input_dim, projection_dim)
        """
            Your optimizer and loss definitions should go here
        """
        self.optimizer = torch.optim.Adam(self.autoencoder_model.parameters(), lr=learning_rate)
        self.loss_fnc = nn.MSELoss()

    def fit(self, x: torch.Tensor) -> None:
        """
        Trains the auto encoder nn on the given data matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should train the auto encoder to minimize the reconstruction error
        please do not forget to put the neural network model into the training mode before training
        """
        self.autoencoder_model.train()  # fnc of nn

        for epoch in range(self.iteration_count):
            self.optimizer.zero_grad()
            x_new = self.autoencoder_model(x)  # fnc of nn
            loss = self.loss_fnc(x_new, x)  # how mcuh was lost during encoding-decoding
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.iteration_count}], Loss: {loss.item():.6f}')

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        After training the nn a given dataset,
        this function uses the learned model to project new data instances
        :param x: the data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        please do not forget to put the neural network model into the evaluation mode before projecting data instances
        """
        self.autoencoder_model.eval()
        with torch.no_grad():
            return self.autoencoder_model.project(x)
