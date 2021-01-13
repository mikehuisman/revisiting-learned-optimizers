import pdb
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Used to decouple direction and norm of tensors which allow us
# to update only the direction (which is what we want for baseline++)
from torch.nn.utils.weight_norm import WeightNorm


from collections import OrderedDict


class SineNetwork(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU())]))
        })
        
        # Output layer
        self.model.update({"out": nn.Linear(40, out_dim)})
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))

    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out
    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        x = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        return x
    
    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                
class ConvBlock(nn.Module):
    """
    Initialize the convolutional block consisting of:
     - 64 convolutional kernels of size 3x3
     - Batch normalization 
     - ReLU nonlinearity
     - 2x2 MaxPooling layer
     
    ...

    Attributes
    ----------
    cl : nn.Conv2d
        Convolutional layer
    bn : nn.BatchNorm2d
        Batch normalization layer
    relu : nn.ReLU
        ReLU function
    mp : nn.MaxPool2d
        Max pooling layer
    running_mean : torch.Tensor
        Running mean of the batch normalization layer
    running_var : torch.Tensor
        Running variance of the batch normalization layer
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    """
    
    def __init__(self, dev, indim=3):
        """Initialize the convolutional block
        
        Parameters
        ----------
        indim : int, optional
            Number of input channels (default=3)
        """
        
        super().__init__()
        self.dev = dev
        self.cl = nn.Conv2d(in_channels=indim, out_channels=64,
                            kernel_size=3)
        self.bn = nn.BatchNorm2d(num_features=64, momentum=1) #momentum=1 is crucial! (only statistics for current batch)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        """Feedforward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 

        Returns
        ----------
        tensor
            The output of the block
        """

        x = self.cl(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)
        return x
    
    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        # Apply conv2d
        x = F.conv2d(x, weights[0], weights[1]) 

        # Manual batch normalization followed by ReLU
        running_mean =  torch.zeros(64).to(self.dev)
        running_var = torch.ones(64).to(self.dev)
        x = F.batch_norm(x, running_mean, running_var, 
                         weights[2], weights[3], momentum=1, training=True)
                              
        x = F.max_pool2d(F.relu(x), kernel_size=2)
        return x
    
    def reset_batch_stats(self):
        """Reset Batch Normalization stats
        """
        
        self.bn.reset_running_stats()
        

class ConvolutionalNetwork(nn.Module):
    """
    Super class for the Conv4 and BoostedConv4 networks.
    
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Feature embedding module of the network (all hidden layers + output layer)
    criterion : loss_fn
        Loss function to use
    in_features : int
        Number of dimensions of embedded inputs
        
    Methods
    ----------
    get_flat_params()
        Returns flattened parameters
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
    
    reset_batch_stats()
        Reset batch normalization stats
    """
            
    def __init__(self, train_classes, eval_classes, criterion, dev):
        """Initialize the conv network

        Parameters
        ----------
        num_classes : int
            Number of classes, which determines the output size
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__()
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.in_features = 3*3*64
        self.criterion = criterion
        
        # Feature embedding module
        self.model = nn.ModuleDict({"features": nn.Sequential(OrderedDict([
            ("conv_block1", ConvBlock(dev=dev, indim=3)),
            ("conv_block2", ConvBlock(dev=dev, indim=64)),
            ("conv_block3", ConvBlock(dev=dev, indim=64)),
            ("conv_block4", ConvBlock(dev=dev, indim=64)),
            ("flatten", nn.Flatten())]))
        })
        
        
    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
                    
    def reset_batch_stats(self):
        """Reset BN stats
        
        Resets the Batch Normalization statistics
        
        """
        
        for m in self.model.modules():
            if isinstance(m, ConvBlock):
                m.reset_batch_stats()        
        
class Conv4(ConvolutionalNetwork):
    """
    Convolutional neural network consisting of four ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss()):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__(train_classes=train_classes, eval_classes=eval_classes, 
                         criterion=criterion, dev=dev)
        self.in_features = 3*3*64
        self.dev = dev
        # Add output layer `out'
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})

        # Set bias weights to 0 of linear layers
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                m.bias = nn.Parameter(torch.zeros(m.bias.size()))
        
    def forward(self, x):
        """Feedforward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 

        Returns
        ----------
        tensor
            The output of the block
        """
        
        features = self.model.features(x)
        out = self.model.out(features)
        return out
    
    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        x = self.model.features.conv_block1.forward_weights(x, weights[0:4])
        x = self.model.features.conv_block2.forward_weights(x, weights[4:8])
        x = self.model.features.conv_block3.forward_weights(x, weights[8:12])
        x = self.model.features.conv_block4.forward_weights(x, weights[12:16])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[16], weights[17])
        return x
    
    def load_state_dict(self, state):
        """Overwritten load_state function
        
        Before loading the state, check whether the dimensions of the output
        layer are matching. If not, create a layer of the output size in the provided state.  

        """
        
        out_key = "model.out.weight"
        out_classes = state[out_key].size()[0]
        if out_classes != self.model.out.weight.size()[0]:
            self.model.out = nn.Linear(in_features=self.in_features,
                                       out_features=out_classes).to(self.dev)
        super().load_state_dict(state)

    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))

    
class PrototypeMatrix(nn.Module):
    """
    Special output layer that learns class representations.
    
    ...
    
    Attributes
    ----------
    prototypes : nn.Parameter
        Parameter-wrapped prototype matrix which contains a column for each class
        
    Methods
    ----------
    forward(x)
        Compute matrix product of x with the prototype matrix     
    """
    
    def __init__(self, in_features, num_classes, dev):
        """Initialize the prototype matrix randomly
        """
        
        super().__init__()
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.rand([in_features, num_classes], device=dev))
    
    def forward(self, x):
        """Apply the prototype matrix to input x
        
        Parameters
        ----------
        x : torch.Tensor
            The input to which we apply our prototype matrix
        
        Returns
        ----------
        tensor
            Result of applying the prototype matrix to input x
        """

        return torch.matmul(x, self.prototypes)
        
class BoostedConv4(ConvolutionalNetwork):
    """
    Convolutional neural network with special output layer.
    This output layer maintains class representations and uses 
    cosine similarity to make predictions. 
    
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    criterion : loss_fn
        Loss function to minimize
     
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers

    Code includes snippets from https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py
    """
    
    def __init__(self, train_classes, eval_classes, dev, criterion=nn.CrossEntropyLoss()):
        """Initialize the conv network

        Parameters
        ----------
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        dev : str
            String identifier of the device to use
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__(train_classes=train_classes, eval_classes=eval_classes, criterion=criterion, dev=dev)
        self.in_features = 3*3*64 #Assuming original image size 84
        self.criterion = criterion
        self.dev = dev

        # Shape is now [batch_size, in_features]
        self.model.update({"out": nn.Linear(in_features=self.in_features, 
                                            out_features=self.train_classes, 
                                            bias=False).to(self.dev)})
        
        WeightNorm.apply(self.model.out, 'weight', dim=0)
        self.scale_factor = 2

        #self.model.update({"out": PrototypeMatrix(in_features=self.in_features, 
        #                                          num_classes=self.train_classes, dev=dev)})

   
    def forward(self, x, test_x=None, test_y=None ,train=True):
        """Feedforward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 

        Returns
        ----------
        tensor
            Scores for every class
        """

        features = self.model.features(x)
        x_norm = torch.norm(features, p=2, dim =1).unsqueeze(1).expand_as(features)
        x_normalized = features.div(x_norm + 0.00001)
        cos_dist = self.model.out(x_normalized) 
        scores = self.scale_factor * cos_dist 
        return scores

        # features = self.model.features(x)
        # # Compute row-wise L2 norm to create a matrix of unit row vectors
        # norm = torch.div(features, torch.reshape(torch.norm(features, dim=1), [-1,1]))

        # # Compute class outputs 
        # class_outputs = self.model.out(norm)
        # return class_outputs

    def load_state_dict(self, state):
        """Overwritten load_state function
        
        Before loading the state, check whether the dimensions of the output
        layer are matching. If not, create a layer of the output size in the provided state.  

        """
        
        out_key = "model.out.weight_g"
        out_classes = state[out_key].size()[0]
        if out_classes != self.model.out.out_features:
            self.model.out = nn.Linear(in_features=self.in_features, 
                                       out_features=out_classes, 
                                       bias=False).to(self.dev)
            WeightNorm.apply(self.model.out, 'weight', dim=0)

        super().load_state_dict(state)


    def freeze_layers(self):
        """Freeze all hidden layers
        """

        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features, 
                                       out_features=self.eval_classes, 
                                       bias=False).to(self.dev)
        WeightNorm.apply(self.model.out, 'weight', dim=0)

        # self.model.out = PrototypeMatrix(in_features=self.in_features, 
        #                                  num_classes=self.eval_classes, dev=self.dev)

