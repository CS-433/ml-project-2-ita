import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

#Creating a class for the dataset (parameters-velocity)

class Fluid_Dataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets =  targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):

        input = self.inputs[idx]
        output = self.targets[idx]
        return input, output
        

def train_solver_autoencoder_epoch(model, device, train_loader, optimizer, epoch, criterion,  vel_space_max, vel_space_min, weights=[1,1],):
    
    """
          train function for the autoencoder + solver architecture

          Args:
              model: model that we are training 
              device: device (CPU or GPU)
              train_loader: utility to efficiently load and organize training data
              optimizer : optimer to update model paramaters
              epoch: int, number of the current epoch
              criterion: loss function
              vel_space_max: max of velocity (output), needed for renormalize the output of the model
              vel_space_min: min of velocity (output), needed for renormalize the output of the model
              weights: list of 2 integers, autoencoder and solver loss weights respectively

         Returns:
              loss_history: list of losses, each related to a batch
              loss_history_enc_dec: list of the losses of autoencoder, each related to a batch
              loss_history_mlp_encoder: list of the losses of the solver, each related to a batch
             
    """
    model.train()

    loss_history = [] #list with train loss for each batch
    loss_history_enc_dec = [] #list with train loss relative to encoder-decoder for each batch
    loss_history_mlp_encoder = [] #list with train loss relative to mlp-encoder for each batch

    for batch_idx, (data, target) in enumerate(train_loader):

        data=data.to(device)
        target=target.to(device)

        output_solver, output_encoder, output_decoder = model.forward(data, target)

        assert output_decoder.shape == target.shape
        assert output_solver.shape == output_encoder.shape

        
        target=target*(1.05*vel_space_max.to(device) - 0.95*vel_space_min.to(device))+0.95*vel_space_min.to(device)  #denormalize
        output_decoder=output_decoder*(1.05*vel_space_max.to(device) - 0.95*vel_space_min.to(device))+0.95*vel_space_min.to(device)

        output_decoder_flattened = output_decoder.flatten(1)   #flatter target and output to compute MSE
        target_fattened = target.flatten(1)

        loss_enc_dec = criterion(target_fattened, output_decoder_flattened)   #loss encoder_decoder
        loss_solver_encoder = criterion(output_solver, output_encoder) #loss solver_encoder


        loss =  weights[0] * loss_enc_dec + weights[1] * loss_solver_encoder #the  loss is the sum of the 2 losses

        optimizer.zero_grad()  # Zero the gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update the weights

        loss_history_enc_dec.append(loss_enc_dec.item())
        loss_history_mlp_encoder.append(loss_solver_encoder.item())
        loss_history.append(loss.item())

    return loss_history, loss_history_enc_dec, loss_history_mlp_encoder


@torch.no_grad()
def validate_solver_autoencoder(model, device, val_loader, criterion, vel_space_max, vel_space_min, weights):

    """
          validate function for the autoencoder + solver architecture

          Args:
              model: model that we are training 
              device: device (CPU or GPU)
              val_loader: utility to efficiently load and organize validation data
              criterion: loss function
              vel_space_max: max of velocity (output), needed for renormalize the output of the model
              vel_space_min: min of velocity (output), needed for renormalize the output of the model
              weights: list of 2 integers, autoencoder and solver loss weights respectively

         Returns:
              test_loss: mean of the loss among the batches used for vaidation
              test_relative_loss: mean of the relative error among the batches used for vaidation
            
    """
    model.eval()
    test_loss = 0
    test_rel_loss = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)

        output_solver, output_encoder, output_decoder = model.forward(data, target)

        assert output_decoder.shape == target.shape
        assert output_solver.shape == output_encoder.shape

        target=target*(1.05*vel_space_max.to(device) - 0.95*vel_space_min.to(device))+0.95*vel_space_min.to(device)   #denormalize
        output_decoder=output_decoder*(1.05*vel_space_max.to(device) - 0.95*vel_space_min.to(device))+0.95*vel_space_min.to(device)

        output_decoder_flattened = output_decoder.flatten(1)   #flatten target and output to compute MSE
        target_fattened = target.flatten(1)

        loss_enc_dec = criterion(target_fattened, output_decoder_flattened)   #loss encoder_decoder
        loss_solver_encoder = criterion(output_solver, output_encoder) #loss solver_encoder


        loss =  weights[0] * loss_enc_dec + weights[1] * loss_solver_encoder #the  loss is the sum of the 2 losses

        test_loss += loss.item() * len(data)  #compute MSE

        output_model_predict = model.predict(data)    #compute output

        output_model_predict=output_model_predict*(1.05*vel_space_max.to(device) - 0.95*vel_space_min.to(device))+0.95*vel_space_min.to(device)

        test_rel_loss += ((torch.norm((output_model_predict-target).view(output_model_predict.shape[0], -1), dim=1, p=2)/torch.norm((target).view(output_model_predict.shape[0], -1), dim=1, p=2)).sum()).item()

    test_loss /= len(val_loader.dataset)
    test_rel_loss /= len(val_loader.dataset)

    return test_loss,test_rel_loss


def run_training_solver_autoencoder(model_solver_autoencoder,num_epochs, lr, batch_size, train_params, train_vel, test_params, test_vel, device,  vel_space_max, vel_space_min, weights = [1,1]):
    
    """
          function used for initialise data loaders, optimizer, criterion, then train and validate

          Args:
              model_solver_autoencoder: model that we want to train and validate 
              num_epochs: number of epochs
              lr: learning rate
              batch_size: batch size
              train_params: params used for trainining
              train_vel: velocity (output) used for training
              test_params: params used for testing
              test_vel: velocity (output) used for testing
              device: device (CPU or GPU)
              vel_space_max: max of velocity (output), needed for renormalize the output of the model
              vel_space_min: min of velocity (output), needed for renormalize the output of the model
              weights: list of 2 integers, autoencoder and solver loss weights respectively

         Returns:
              model_solver_autoencoder:
              train_loss_history: list of train losses for each epoch
              val_loss_history: list of validation losses for each epoch
              val_rel_loss_history: list of validation relative errors for each epoch
              train_loss_enc_dec_history: list of train loss relative to the autoencoder for each epoch
              train_loss_solver_enc_history: list of train loss relative to the solver for each epoch
            
    """
    train_vel_DataSet = Fluid_Dataset(train_params, train_vel)
    test_vel_DataSet = Fluid_Dataset(test_params, test_vel)

    train_loader = torch.utils.data.DataLoader(
        train_vel_DataSet,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        test_vel_DataSet,
        batch_size=batch_size,
    )

    # ===== Model, Optimizer and Criterion =====
    model_solver_autoencoder = model_solver_autoencoder.to(device=device)

    optimizer  = torch.optim.Adam(
        model_solver_autoencoder.parameters(),
        lr=lr,
        weight_decay = 1e-6
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    criterion = torch.nn.functional.mse_loss

    # ===== Train Model =====
    train_loss_history = []
    train_loss_enc_dec_history = []
    train_loss_solver_enc_history = []
    val_loss_history = []
    val_rel_loss_history = []


    for epoch in range(1, num_epochs + 1):

        train_loss, train_loss_enc_dec, train_loss_solver_enc = train_solver_autoencoder_epoch(
           model_solver_autoencoder, device, train_loader, optimizer, epoch, criterion,vel_space_max, vel_space_min, weights, 
        )

        train_loss_history.append(np.mean(np.array(train_loss)))
        train_loss_enc_dec_history.append(np.mean(np.array(train_loss_enc_dec)))
        train_loss_solver_enc_history.append(np.mean(np.array(train_loss_solver_enc)))

        val_loss,val_rel_loss = validate_solver_autoencoder(model_solver_autoencoder, device, val_loader, criterion, vel_space_max, vel_space_min, weights)
        val_loss_history.append(val_loss)
        val_rel_loss_history.append(val_rel_loss)

        scheduler.step(val_loss)

    return model_solver_autoencoder, train_loss_history,  val_loss_history, val_rel_loss_history, train_loss_enc_dec_history, train_loss_solver_enc_history




#SEQUENTIAL
def train_autoencoder_epoch(autoencoder, device, train_loader, optimizer, criterion, vel_space_max, vel_space_min):
    """
          train function for the autoencoder 

          Args:
              model: model that we are training 
              device: device (CPU or GPU)
              train_loader: utility to efficiently load and organize training data
              optimizer : optimer to update model paramaters
              criterion: loss function
              vel_space_max: max of velocity (output), needed for renormalize the output of the model
              vel_space_min: min of velocity (output), needed for renormalize the output of the model
              

         Returns:
              loss_history: list of losses, each related to a batch
            
    """
    autoencoder.train()

    loss_history = []

    for batch_idx, (data, target) in enumerate(train_loader):


        data=data.to(device) 
        target=target.to(device) 

        output_encoder, output_decoder = autoencoder.forward(target)

        assert output_decoder.shape == target.shape

        target=target*(1.05*vel_space_max.to(device) - 0.95*vel_space_min.to(device))+0.95*vel_space_min.to(device)
        output_decoder=output_decoder*(1.05*vel_space_max.to(device) - 0.95*vel_space_min.to(device))+0.95*vel_space_min.to(device)

        output_decoder_flattened = output_decoder.flatten(1)
        target_fattened = target.flatten(1)

        loss = criterion(output_decoder_flattened, target_fattened)

        optimizer.zero_grad()  # Zero the gradients

        loss.backward()        # Backpropagation
        optimizer.step()       # Update the weights

        loss_history.append(loss.item())

    return loss_history


@torch.no_grad()
def validate_autoencoder(autoencoder, device, val_loader, criterion, vel_space_max, vel_space_min):
    
    """
          validate function for the autoencoder 

          Args:
              model: model that we are training 
              device: device (CPU or GPU)
              val_loader: utility to efficiently load and organize validation data
              criterion: loss function
              vel_space_max: max of velocity (output), needed for renormalize the output of the model
              vel_space_min: min of velocity (output), needed for renormalize the output of the model

         Returns:
              test_loss: mean of the loss among the batches used for vaidation
            
            
    """
    autoencoder.eval()  # Important set model to eval mode (affects dropout, batch norm etc)
    test_loss = 0

    for data, target in val_loader:

        data, target = data.to(device), target.to(device)
        output_encoder, output_decoder = autoencoder.forward(target)
        assert output_decoder.shape == target.shape

        target=target*(1.05*vel_space_max.to(device) - 0.95*vel_space_min.to(device))+0.95*vel_space_min.to(device)
        output_decoder=output_decoder*(1.05*vel_space_max.to(device) - 0.95*vel_space_min.to(device))+0.95*vel_space_min.to(device)

        output_flattened = output_decoder.flatten(1)
        target_flattened = target.flatten(1)
        test_loss += criterion(output_flattened, target_flattened).item() * len(data)


    test_loss /= len(val_loader.dataset)

    return test_loss


def run_training(model, num_epochs, lr, batch_size, train_params, train_vel, test_params, test_vel, vel_space_max, vel_space_min, device="cuda"):

    """
          function used for initialise data loaders, optimizer, criterion, then train and validate

          Args:
              model: model that we want to train and validate 
              num_epochs: number of epochs
              lr: learning rate
              batch_size: batch size
              train_params: params used for trainining
              train_vel: velocity (output) used for training
              test_params: params used for testing
              test_vel: velocity (output) used for testing
              vel_space_max: max of velocity (output), needed for renormalize the output of the model
              vel_space_min: min of velocity (output), needed for renormalize the output of the model
              device: device (CPU or GPU)
              
      

         Returns:
              model: the model trained
              train_loss_history: list of train losses for each epoch
              val_loss_history: list of validation losses for each epoch
            
    """

    train_vel_DataSet = Fluid_Dataset(train_params, train_vel)
    test_vel_DataSet = Fluid_Dataset(test_params, test_vel)

    train_loader = torch.utils.data.DataLoader(
        train_vel_DataSet,
        batch_size=batch_size,
        shuffle=True,  # Can be important for training
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        # num_workers=2,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        test_vel_DataSet,
        batch_size=batch_size,
    )

    # ===== Model, Optimizer and Criterion =====
    model = model.to(device=device)



    optimizer  = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay = 1e-6
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    criterion = torch.nn.functional.mse_loss
    # ===== Train Model =====
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, num_epochs + 1):

        train_loss = train_autoencoder_epoch(
           model, device, train_loader, optimizer, criterion, vel_space_max, vel_space_min
        )
        train_loss_history.append(np.mean(train_loss))

        val_loss= validate_autoencoder(model, device, val_loader, criterion, vel_space_max, vel_space_min)
        val_loss_history.append(val_loss)

        scheduler.step(val_loss)

    return model, train_loss_history,  val_loss_history


#solver
def train_solver_epoch(solver, device, train_loader, optimizer, criterion, autoencoder_trained):
    """
          train function for the solver 

          Args:
              model: model that we are training 
              device: device (CPU or GPU)
              train_loader: utility to efficiently load and organize training data
              optimizer : optimer to update model paramaters
              criterion: loss function
              autoencoder_trained: model of autoencoder already trained
              

         Returns:
              loss_history: list of losses, each related to a batch
            
    """
    solver.train()

    loss_history = []

    for batch_idx, (data, target) in enumerate(train_loader):

        data=data.to(device)
        target=target.to(device)

        output_solver = solver.forward(data)

        autoencoder_trained.eval()
        with torch.no_grad():
          output_encoder =  autoencoder_trained.encoder(target) #output of encoder

        loss = criterion(output_encoder, output_solver)

        optimizer.zero_grad()  # Zero the gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update the weights

        loss_history.append(loss.item())

    return loss_history


@torch.no_grad()
def validate_solver(solver, device, val_loader, criterion, autoencoder_trained):
    
    """
          validate function for the solver 

          Args:
              solver: model that we are training 
              device: device (CPU or GPU)
              val_loader: utility to efficiently load and organize validation data
              criterion: loss function
              autoencoder_trained: model of autoencoder already trained
              

         Returns:
              test_loss: mean of the loss among the batches used for vaidation
            
            
    """
    solver.eval() 

    test_loss = 0

    for data, target in val_loader:

        data, target = data.to(device), target.to(device)
        output_solver = solver.forward(data)

        autoencoder_trained.eval()
        output_encoder =  autoencoder_trained.encoder(target)

        test_loss += criterion(output_encoder, output_solver).item() * len(data)

    test_loss /= len(val_loader.dataset)

    return test_loss

def run_training_solver(autoencoder_trained, model, num_epochs, lr, batch_size, train_params, train_vel, test_params, test_vel, device="cuda"):
    """
          function used for initialise data loaders, optimizer, criterion, then train and validate

          Args:
              autoencoder_trained: model of the autoencoder already trained
              model: model of the solver that we want to train and validate 
              num_epochs: number of epochs
              lr: learning rate
              batch_size: batch size
              train_params: params used for trainining
              train_vel: velocity (output) used for training
              test_params: params used for testing
              test_vel: velocity (output) used for testing
              device: device (CPU or GPU)
              
      

         Returns:
              model: the model of the solver trained 
              train_loss_history: list of train losses for each epoch
              val_loss_history: list of validation losses for each epoch
            
    """
    train_vel_DataSet = Fluid_Dataset(train_params, train_vel)
    test_vel_DataSet = Fluid_Dataset(test_params, test_vel)

    train_loader = torch.utils.data.DataLoader(
        train_vel_DataSet,
        batch_size=batch_size,
        shuffle=True,  # Can be important for training
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        # num_workers=2,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        test_vel_DataSet,
        batch_size=batch_size,
    )

    # ===== Model, Optimizer and Criterion =====
    model = model.to(device=device)



    optimizer  = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay = 1e-6
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    criterion = torch.nn.functional.mse_loss
    # ===== Train Model =====
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, num_epochs + 1):

        train_loss = train_solver_epoch(
           model, device, train_loader, optimizer, criterion,  autoencoder_trained
        )
        train_loss_history.append(np.mean(train_loss))

        val_loss= validate_solver(model, device, val_loader, criterion, autoencoder_trained)
        val_loss_history.append(val_loss)

        scheduler.step(val_loss)

    return model, train_loss_history,  val_loss_history

