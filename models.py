import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


#model of the solver
class Solver(torch.nn.Module):
    def __init__(self, L=4, K=256, dim_reduced=128 ,activation="gelu"):
      """
         fully connected solver model

          Args:
              L: number of hidden layers
              K: number of neurons per
              dim_reduced: size of the last layer
              activation: string encoding the activation function used in the model
          """

      super(Solver, self).__init__()

      self.activation=torch.nn.ReLU if activation=='relu' else torch.nn.GELU
      self.solver = torch.nn.Sequential(
            *[layer for i in range(L) for layer in [
                torch.nn.Linear(3 if i == 0 else K, K),
                torch.nn.LayerNorm(K),
                self.activation(),
            ]],
            torch.nn.Linear(K, dim_reduced),
            torch.nn.LayerNorm(dim_reduced),
            self.activation()
        )
      
    def forward(self, par):
        output = self.solver(par)
        return output

#model of the autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self, activation, L_enc_dec,  dim_reduced=128, dim_1=39, dim_2=16):
        """
         fully connected autoencoder model

          Args:
              activation: activation function ("relu" or "gelu")
              L_enc_dec: int, number layers of the encoder (=number of layers of the decoder)
              dim_reduced : int,  the size to which we want to reduce
              dim_1: int,  first dimension  of the output matrix (39)
              dim_2: int, second dimension of the output matrix (16)

          """

        super(Autoencoder, self).__init__()

        #the depth of each layer (i.e., the number of neurons per layer) decrease proportionally, 
        # r is the reduction factor between consecutive layers
        r = (dim_reduced / (dim_1*dim_2)) ** (1 / L_enc_dec)

        # dims contains the values of the depth of each layer of the encoder
        dims = [round(dim_1*dim_2 * (r ** i)) for i in range(L_enc_dec)] + [dim_reduced]

        self.activation=torch.nn.ReLU if activation=='relu' else torch.nn.GELU

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            *[layer for i in range(L_enc_dec) for layer in [
                torch.nn.Linear(dims[i], dims[i+1]),
                torch.nn.LayerNorm(dims[i+1]),
                self.activation()
            ]])

        self.decoder = torch.nn.Sequential(
            *[layer for i in range(L_enc_dec-1) for layer in [
                torch.nn.Linear(dims[L_enc_dec-i], dims[L_enc_dec-i-1]),
                torch.nn.LayerNorm(dims[L_enc_dec-i-1]),
                self.activation()
            ]],
            torch.nn.Linear(dims[1], dims[0]), # dim output = (batch_size, dim_1 * dim_2 )
            #torch.nn.Sigmoid(),
            torch.nn.Unflatten(1, (dim_1, dim_2)), # dim_output = (batch_size, dim_1, dim_2)
            )

    def forward(self, out):
        encoder_output = self.encoder(out)
        output = self.decoder(encoder_output)

        return encoder_output, output


class SolverPlusMLPAutoencoder(torch.nn.Module):
    def __init__(self, activation, L_enc_dec, L_l=4, dim_reduced=128, K=256, dim_1=39, dim_2=16):

        """
          Solver MLP + Autoencoder MLP model

          Args:
              activation: activation function ("relu" or "gelu")
              L_enc_dec: int, number layers of the encoder (=number of layers of the decoder)
              L_l: int, number of hidden layers in the Solver
              dim_reduced : int,  the size to which we want to reduce
              K: int, depth of layers in the Solver
              dim_1: int,  first dimension  of the velocity matrix (39)
              dim_2: int, second dimension of the velocity matrix (16)
        """
        super(SolverPlusMLPAutoencoder, self).__init__()

        #Solver
        self.solver = Solver(L=L_l, K=K, dim_reduced=dim_reduced, activation=activation).solver

        #Encoder
        self.encoder = Autoencoder(activation, L_enc_dec, dim_reduced=dim_reduced, dim_1=dim_1, dim_2=dim_2).encoder

        #Decoder
        self.decoder = Autoencoder(activation, L_enc_dec, dim_reduced=dim_reduced, dim_1=dim_1, dim_2=dim_2).decoder

    def forward(self, par, out):

        output_solver = self.solver(par)  # (batch_size, dim_reduced)
        output_encoder = self.encoder(out)  # (batch_size, dim_reduced)
        output_decoder = self.decoder(output_encoder) # (batch_size, dim_1, dim_2)
        return output_solver, output_encoder, output_decoder

    def predict(self, par):
        output_solver = self.solver(par)  # (batch_size, dim_reduced)
        output_model_predict = self.decoder(output_solver)  # (batch_size,  dim_1, dim_2)
        return output_model_predict



class SolverPlusConvAutoencoder(torch.nn.Module):
    def __init__(self, activation, pooling, K_comb, L_c, L_l, dim_1=39, dim_2=16, dim_reduced=128, K=256):


        """
        MLP + autoencoder model, where convolutional layers are reduced in size by a pooling  (H,W) -> (H/2, W/2)
        Args:
            activation: activation function ("relu" or "gelu")
            pooling: pooling ("avg" or "max")
            K_comb: array of 3 elements, indicationg kernel size, stride and padding of convolutional layers
            L_c: int, number of convolutional layers
            L_l: int, number of dense layers after convolutional layers
            dim_1: int,  first dimension  of the output matrix (39)
            dim_2: int, second dimension of the output matrix (16)
            dim_reduced : int,  the size to which we want to reduce
            K: int, depth of layers in the solver model

        """
        super(SolverPlusConvAutoencoder, self).__init__()

        #compute the dimension of the output of the decoder and the dimension of the output of each convolutional layers, based
        #on the number of convolutional layers

        if L_c == 1:
            dim_post_encoder = 2432
            tuple_dec=[(16,19,8),(39,16)]
        elif L_c == 2:
            dim_post_encoder = 1152
            tuple_dec=[(32,9,4),(19,8),(39,16)]
        elif L_c == 3:
            dim_post_encoder = 512
            tuple_dec=[(64,4,2),(9,4),(19,8),(39,16)]

        self.solver = Solver(L=4, K=K, dim_reduced=dim_reduced, activation=activation)

        self.activation=torch.nn.ReLU if activation=='relu' else torch.nn.GELU
        self.pooling=torch.nn.MaxPool2d if pooling=='max' else torch.nn.AvgPool2d
        initial_h=dim_1
        initial_w=dim_2

        #Encoder
        self.encoder = torch.nn.Sequential(
            *[layer for i in range(L_c) for layer in [
                torch.nn.Conv2d(1 if i == 0 else 2**(i-1)*16, 2**(i)*16, kernel_size=K_comb[0], stride=K_comb[1], padding=K_comb[2]),
                torch.nn.LayerNorm([2**(i)*16,initial_h//(2**i),initial_w//(2**i)]), 
                self.activation(),
                self.pooling(2,2)
            ]],
            torch.nn.Flatten(),
            *[layer for i in range(L_l) for layer in [
                torch.nn.Linear(dim_post_encoder if i==0 else dim_reduced*(L_l-i+1), dim_reduced*(L_l-i)),
                torch.nn.LayerNorm(dim_reduced*(L_l-i)),
                self.activation()
            ]],

        )
        
        #Decoder
        self.decoder = torch.nn.Sequential(
            *[layer for i in range(L_l-1,-1,-1) for layer in [
                torch.nn.Linear(dim_reduced*(L_l-i), dim_post_encoder if i==0 else dim_reduced*(L_l-i+1)),
                torch.nn.LayerNorm(dim_post_encoder if i==0 else dim_reduced*(L_l-i+1)),
                self.activation()
            ]],
            torch.nn.Unflatten(1, tuple_dec[0]),
            *[layer for i in range(L_c-1) for layer in [
                torch.nn.Upsample(size=tuple_dec[i+1]),
                torch.nn.ConvTranspose2d(2**(L_c-i-1)*16, 2**(L_c-i-2)*16, kernel_size=K_comb[0], stride=K_comb[1], padding=K_comb[2]),
                torch.nn.LayerNorm([2**(L_c-i-2)*16,*tuple_dec[i+1]]), 
                self.activation(),
            ]],
            torch.nn.Upsample(size=tuple_dec[-1]),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=K_comb[0], stride=K_comb[1], padding=K_comb[2]),
            # torch.nn.Sigmoid()
        )



    def forward(self, par, out):
        output_solver = self.solver(par)  # (batch_size, dim_reduced)

        out_squeezed = out.unsqueeze(1) #(batch_size x 39 x 16) -> #(batch_size x 1 x 39 x 16)

        output_encoder = self.encoder(out_squeezed)  # (batch_size, dim_reduced)

        output_decoder = self.decoder(output_encoder) # (batch_siz, 1, dim_1, dim_2)
        output_decoder=output_decoder.squeeze(1)  # (batch_siz, dim_1, dim_2)

        return output_solver, output_encoder, output_decoder

    def predict(self, par):
        output_solver = self.solver(par)  # (dim_1_reduced * dim_2_reduced)
        output_predict = self.decoder(output_solver)  # (batch_size, 1, dim_1, dim_2)
        output_predict=output_predict.squeeze(1)
        return output_predict