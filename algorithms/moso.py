import torch
import torch.nn as nn

from .algorithm import Algorithm
from .modules.shell import MetaLearner, get_init_info, INP_TO_CODE
from .modules.utils import get_loss_and_grads, put_on_device, update,\
                           preprocess_grad_loss, new_weights, deploy_on_task,\
                           eval_model, set_weights
    
    
class MOSO(Algorithm):
    """Mimicking One-Step Optimizer
    
    MOSO attempts to mimick hand-crafted optimizers within
    a single feed-forward pass. 
    Suppose you use regular SGD as optimizer, and after 100
    optimizations steps, you obtain weights w1, ..., wn. 
    MOSO will attempt to obtain these weights from a single 
    feed-forward pass on the initial loss gradient information.
    
    ...

    Attributes
    ----------
    baselearner_fn : constructor function
        Constructor function for the base-learner
    baselearner_args : dict
        Dictionary of keyword arguments for the base-learner
    baselearner : nn.Module
        Baselearner model. We keep this as attribute to maintain the weights 
        from previous tasks
    metalearner : OSOMetaLearner
        The meta-learning network that proposes one-step updates to baselearner
        weights
    opt_fn : constructor function
        Constructor function for the optimizer to use
    meta_opt : torch.optim
        Meta optimizer used to make updates to MOSO
    hcopt_fn : constructor function
        Constructor for the base-learner optimizer
    T : int
        Number of update steps to parameters per task
    train_batch_size : int
        Indicating the size of minibatches that are sampled from meta-train tasks
    test_batch_size : int
        Size of batches to sample from meta-[val/test] tasks
    meta_batch_size : int
        Used batch size for training MOSO
    lr : float
        Learning rate for the optimizer
    hcopt_lr : float
        Learning rate for the base-learner optimizer
    validation : boolean
        Whether this model should use meta-validation
    cpe : int
        Optimization checkpoints per episode (how often to reconsider best
        weight found by hand-crafted optimizer)
    dev : str
        Device identifier
    trainable : boolean
        Whether it can train on meta-trrain tasks
    episodic : boolean
        Whether to sample tasks or mini batches for training
    """
    
    def __init__(self, input_type, layers, hcopt_fn, hcopt_lr, act, 
                 cpe, meta_batch_size, **kwargs):
        """Set all (inherited) attributes
        
        Attributes
        ----------
        input_type : str
            Specifier of the input type (raw_grads, raw_loss_grads, proc_loss_grads,
            maml). Determines the input to the meta-learner network
        layers : list
            List of integers corresponding to the number of neurons per hidden/output layer
            e.g., [5,5,1] is a neural network [input -> hidden 5 -> hidden 5 -> output 1]
        hcopt_fn : torch.optim
            Constructor for hand-crafted optimizer to mimick
        hcopt_lr : float
            Learning rate for the base-learner optimizer
        act : act_fn
            Activation function to user for meta-learner
        cpe : int
            How often to recompute best weights on support set
            during the optimization path of the hand-crafted
            optimizer
        meta_batch_size : int
            Used batch size for training MOSO
        **kwargs : dict
            Ignored arguments
        """
        
        super().__init__(**kwargs)
        self.input_type = input_type
        self.input_code = INP_TO_CODE[input_type]
        self.cpe = cpe
        self.meta_batch_size = meta_batch_size
        self.hcopt_fn = hcopt_fn
        self.hcopt_lr = hcopt_lr
        
        # Create random base-learner just to get batch norm layer names
        baselearner = self.baselearner_fn(**self.baselearner_args)

        # Create meta-learner network and meta-optimizer
        self.metalearner = MetaLearner(input_type=input_type, layers=layers,
                                       activation=act).to(self.dev)
        self.meta_opt = self.opt_fn(self.metalearner.parameters(), lr=self.lr)
        
        
        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]
        
        
    def train(self, train_x, train_y, test_x, test_y):
        """Train on a given task
        
        Run hand-crafted optimizer on support set for T 
        epochs and take the final best weights as target.
        Inputs for the meta-learner are then the initial loss
        and gradients (processed) and targets are the best weights.
        
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

        # Create random base-learner
        baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        # Put baselearner in train mode
        baselearner.train()
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        
        # Initialize hand-crafted optimizer for base-learner (to reset stats from,
        # e.g., previous train call)
        hcopt = self.hcopt_fn(baselearner.parameters(), lr=self.hcopt_lr)
        # Initial base-learner weights
        init_weights = [p.clone().detach() for p in baselearner.parameters()]
        
        # Construct meta input matrix
        X, _ = get_init_info(baselearner, train_x, train_y, input_code=self.input_code)
            
        # Maintain best weights for the baselearner so far
        best_loss = float("inf")
        best_weights = [p.clone().detach() for p in baselearner.parameters()]
        
        # Compute after how many epochs, the best weights should be reconsidered
        # to get @self.cpe reconsiderations in total over @self.T epochs
        val_after = self.T//self.cpe
        
        # Make T updates with hand-crafted optimizer
        for t in range(self.T):
            # Perform single update 
            update(baselearner, hcopt, train_x, train_y)
            
            # Reconsider best weights and loss so far
            if val_after != 0:
                if (t + 1) % val_after == 0:
                    best_weights, best_loss = new_weights(baselearner, best_weights, best_loss, 
                                                          test_x, test_y, operator=self.operator,
                                                          ls=True)
        
        # Compute targets (final_param - orig_param) for the meta-learner network
        # for all base-learner parameters
        Y = torch.cat([(p - op).reshape([-1, 1]) for op, p\
                       in zip(init_weights, best_weights)]).detach()
        
        
        T = len(Y) // self.meta_batch_size 
        # Train meta-learner model on X, Y
        deploy_on_task(self.metalearner, self.meta_opt, X, Y, 
                       None, None, T, self.meta_batch_size, cpe=self.cpe,
                       init_score=float("inf"), operator=min)        
        
    def evaluate(self, train_x, train_y, test_x, test_y):
        """Evaluate on a given task
        
        Use the support set (train_x, train_y) and/or 
        the query set (test_x, test_y) to evaluate the performance of 
        the baselearner after proposed updates by MOSO.
        
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
        
        Returns
        ----------
        loss
            Loss or accuracy on the test set
        """
        
        # Create random base-learner
        baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        # Put baselearner in evaluation mode
        baselearner.eval()
        # Put tensors on appropriate devices
        train_x, train_y, test_x, test_y = put_on_device(self.dev,
                                                         [train_x, train_y,
                                                          test_x, test_y])
        # Construct meta input matrix
        X, _ = get_init_info(baselearner, train_x, train_y, input_code=self.input_code)
        # Compute weight update proposals 
        updates = self.metalearner(X)
        # Compute new weights with the proposed updates
        fast_weights = [p.clone() for p in baselearner.parameters()]
        
        # Keep lower-bound and upper-bound as moving window 
        # on flat updates
        lb = 0
        ub = 0
        for i in range(len(fast_weights)):
            num_params = fast_weights[i].numel()
            ub += num_params
            fast_weights[i] = fast_weights[i] + updates[lb:ub].reshape(fast_weights[i].shape)
            lb += num_params
            
        # Load new weights into the base-learner
        set_weights(baselearner, fast_weights, self.keys, self.bn_keys)
        
        # Evaluate loss on query set
        loss = eval_model(baselearner, test_x, test_y, operator=self.operator)
        return loss
        
    def dump_state(self):
        """Return the state of the meta-learner
        
        Returns
        ----------
        state_dict
            State dictionary of the meta-learner
        """
        
        return self.metalearner.state_dict()
    
    def load_state(self, state):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : state_dict
            State dictionary of the meta-learner
        """
        
        self.metalearner.load_state_dict(state)    
        