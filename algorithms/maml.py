import torch

from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy


class MAML(Algorithm):
    """Model-Agnostic Meta-Learning
    
    Meta-learning algorithm that attempts to obtain a good common 
    initialization point (base-learner parameters) across tasks.
    From this initialization point, we want to be able to make quick
    task-specific updates to achieve good performance from just few
    data points.
    Our implementation performs a single step of gradient descent
    
    ...

    Attributes
    ----------
    baselearner_fn : constructor function
        Constructor function for the base-learner
    baselearner_args : dict
        Dictionary of keyword arguments for the base-learner
    opt_fn : constructor function
        Constructor function for the optimizer to use
    T : int
        Number of update steps to parameters per task
    train_batch_size : int
        Indicating the size of minibatches that are sampled from meta-train tasks
    test_batch_size : int
        Size of batches to sample from meta-[val/test] tasks
    lr : float
        Learning rate for the optimizer
    validation : boolean
        Whether this model should use meta-validation
    dev : str
        Device identifier
    trainable : boolean
        Whether it can train on meta-trrain tasks
    episodic : boolean
        Whether to sample tasks or mini batches for training
        
    Methods
    -------
    train(train_x, train_y, test_x, test_y)
        Perform a single training step on a given task
    
    evaluate(train_x, train_y, test_x, test_y)
        Evaluate the performance on the given task
        
    dump_state()
        Dump the meta-learner state s.t. it can be loaded again later
        
    load_state(state)
        Set meta-learner state to provided @state 
    """
    
    def __init__(self, T_test, base_lr, second_order, grad_clip=None, meta_batch_size=1, **kwargs):
        """Initialization of model-agnostic meta-learner
        
        Parameters
        ----------
        T_test : int
            Number of updates to make at test time
        base_lr : float
            Learning rate for the base-learner 
        second_order : boolean
            Whether to use second-order gradient information
        grad_clip : float
            Threshold for gradient value clipping
        meta_batch_size : int
            Number of tasks to compute outer-update
        **kwargs : dict
            Keyword arguments that are ignored
        """
        
        super().__init__(**kwargs)
        self.base_lr = base_lr
        self.grad_clip = grad_clip        
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size
        self.T_test = T_test
        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        
        # Get random initialization point for baselearner
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in self.baselearner.parameters()]

        # Store gradients across tasks
        self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in self.initialization]

        # Enable gradient tracking for the initialization parameters
        for p in self.initialization:
            p.requires_grad = True
                
        # Initialize the meta-optimizer
        self.optimizer = self.opt_fn(self.initialization, lr=self.lr)
        
        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]
            
    def _fast_weights(self, params, gradients):
        """Compute task-specific weights using the gradients
        
        Apply a single step of gradient descent using the provided gradients
        to compute task-specific, or equivalently, fast, weights.
        
        Parameters
        ----------
        params : list
            List of parameter tensors
        gradients : list
            List of torch.Tensor variables containing the gradients per layer
        """
        
        # Clip gradient values between (-10, +10)
        if not self.grad_clip is None:
            gradients = [torch.clamp(p, -self.grad_clip, +self.grad_clip) for p in gradients]
        
        fast_weights = [params[i] - self.base_lr * gradients[i]\
                        for i in range(len(gradients))]
        
        return fast_weights
    
    def _deploy(self, train_x, train_y, test_x, test_y, train_mode):
        """Run DOSO on a single task to get the loss on the query set
        
        1. Evaluate the base-learner loss and gradients on the support set (train_x, train_y)
        using our initialization point.
        2. Make a single weight update based on this information.
        3. Evaluate and return the loss of the fast weights (initialization + proposed updates)
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        train_mode : boolean
            Whether we are in training mode or test mode

        Returns
        ----------
        test_loss
            Loss of the base-learner on the query set after the proposed
            one-step update
        """
        
        fast_weights = [p.clone() for p in self.initialization]
        
        if train_mode:
            T = self.T
        else:
            T = self.T_test

        for step in range(T):     
            _, grads = get_loss_and_grads(self.baselearner, train_x, train_y, 
                                          weights=fast_weights, 
                                          create_graph=self.second_order,
                                          retain_graph=T > 1 or self.second_order,
                                          flat=False)

            fast_weights = self._fast_weights(params=fast_weights, gradients=grads)

        # Get and return performance on query set
        test_preds = self.baselearner.forward_weights(test_x, fast_weights)
        test_loss = self.baselearner.criterion(test_preds, test_y)
        return test_loss, test_preds
    
    def train(self, train_x, train_y, test_x, test_y):
        """Train on a given task
        
        Start with the common initialization point and perform a few
        steps of gradient descent from there using the support set
        (rain_x, train_y). Observe the error on the query set and 
        propagate the loss backwards to update the initialization.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        """ 
        
        # Put baselearner in training mode
        self.baselearner.train()
        self.task_counter += 1

        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, _ = self._deploy(train_x, train_y, test_x, test_y, True)

        # Propagate the test loss backwards to update the initialization point
        test_loss.backward()
            
        # Clip gradients
        if not self.grad_clip is None:
            for p in self.initialization:
                p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)

        self.grad_buffer = [self.grad_buffer[i] + self.initialization[i].grad for i in range(len(self.initialization))]
        self.optimizer.zero_grad()

        if self.task_counter % self.meta_batch_size == 0: 
            # Copy gradients from self.grad_buffer to gradient buffers in the initialization parameters
            for i, p in enumerate(self.initialization):
                p.grad = self.grad_buffer[i]
            self.optimizer.step()
            
            self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in self.initialization]
            self.task_counter = 0
            self.optimizer.zero_grad()

    def evaluate(self, train_x, train_y, test_x, test_y):
        """Evaluate on a given task
        
        Use the support set (train_x, train_y) and/or 
        the query set (test_x, test_y) to evaluate the performance of 
        the model.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        """
        
        # Put baselearner in evaluation mode
        self.baselearner.eval()
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        # Compute the test loss after a single gradient update on the support set
        test_loss, preds = self._deploy(train_x, train_y, test_x, test_y, False)

        if self.operator == min:
            return test_loss.item()
        else:
            # Turn one-hot predictions into class preds
            preds = torch.argmax(preds, dim=1)
            return accuracy(preds, test_y)
    
    def dump_state(self):
        """Return the state of the meta-learner
        
        Returns
        ----------
        initialization
            Initialization parameters
        """
        return [p.clone().detach() for p in self.initialization]
    
    def load_state(self, state):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True
        