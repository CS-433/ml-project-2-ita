import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot(ax, train_loss_history, val_loss_history, num_epochs):
    """
          Plot on ax the train loos, the validation loss given the number of epochs

          Args:
              ax: axis on which to plot
              train_loss_history: list of train losses
              val_loss_history: list of validation losses
              num_epochs: number of epochs
          """
    

    t = np.arange(1, num_epochs + 1)

    ax.semilogy(t, train_loss_history, label="Train loss")
    ax.semilogy(t, val_loss_history, label="Val loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True) 