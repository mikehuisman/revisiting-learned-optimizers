from .algorithm import Algorithm
from .modules.utils import eval_model, get_batch, new_weights, update,\
                   put_on_device, deploy_on_task

class FineTuning(Algorithm):
    """Transfer-learning model based on pre-training and fine-tuning
    
    Model that pre-trains on batches from all meta-training data,
    and finetunes only the last layer when presented with tasks at
    meta-validation/meta-test time.
    
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
    cpe : int
        Checkpoints per episode (# times we recompute new best weights)
    dev : str
        Device identifier
    trainable : boolean
        Whether it can train on meta-trrain tasks
    frozen : boolean
        Flag whether the weights of all except the final layer has been frozen
        
    Methods
    -------
    _freeze_layers()
        Freeze all hidden layers
        
    train(train_x, train_y, **kwargs)
        Train on a given batch of data
        
    evaluate(train_x, train_y, test_x, test_y)
        Evaluate the performance on the given task
        
    dump_state()
        Dumps the model state
        
    load_state(state)
        Loads the given model state    
    """
    
    def __init__(self, cpe, **kwargs):
        """
        Call parent constructor function to inherit and set attributes
        Create a model that will be used throughout. Set frozen state to be false.
        
        Parameters
        ----------
        cpe : int
            Number of times the best weights should be reconsidered in an episode
        """
        
        super().__init__(**kwargs)
        self.cpe = cpe
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.baselearner.train()
        self.val_learner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.val_learner.load_state_dict(self.baselearner.state_dict())
        self.val_learner.train()

        #print(self.baselearner.model.out.prototypes.device); import sys; sys.exit()
        self.optimizer = self.opt_fn(self.baselearner.parameters(), lr=self.lr)
        self.episodic = False
                
    def train(self, train_x, train_y, **kwargs):
        """Train on batch of data
        
        Train for a single step on the support set data
        
        Parameters
        ----------
        train_x : torch.Tensor
            Tensor of inputs
        train_y : torch.Tensor
            Tensor of ground-truth outputs corresponding to the inputs
        **kwargs : dict
            Ignore other parameters
        """
        
        self.baselearner.train()
        self.val_learner.train()
        #print("Bias nodes in output layer:", self.baselearner.model.out.bias.size())
        train_x, train_y = put_on_device(self.dev, [train_x, train_y])
        update(self.baselearner, self.optimizer, train_x, train_y)

        
    def evaluate(self, train_x, train_y, test_x, test_y):
        """Evaluate the model on a task
        
        Fine-tune the last layer on the support set of the task and
        evaluate on the query set.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Tensor of inputs
        train_y : torch.Tensor
            Tensor of ground-truth outputs corresponding to the inputs
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Ground-truth outputs of the query set
        
        Returns
        ----------
        test_loss
            Loss (float) on the query set
        """
        
        # CHeck if the mode has changed (val_learner.training=True during training). 
        # If so, set weights of validation learner to those of base-learner
        change = self.val_learner.training != False
        if change:
            self.val_learner.load_state_dict(self.baselearner.state_dict())
            
        # Check if mode is changed
        self.val_learner.eval()
        
        # If hidden layers not frozen yet, make em cold!
        self.val_learner.freeze_layers()
        
        val_optimizer = self.opt_fn(self.val_learner.parameters(), self.lr)
        
        # Put on the right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev, 
                                            [train_x, train_y, 
                                             test_x, test_y])

        # Train on support set and get loss on query set
        test_loss = deploy_on_task(
                        model=self.val_learner, 
                        optimizer=val_optimizer,
                        train_x=train_x, 
                        train_y=train_y, 
                        test_x=test_x, 
                        test_y=test_y, 
                        T=self.T, 
                        test_batch_size=self.test_batch_size,
                        cpe=self.cpe,
                        init_score=self.init_score,
                        operator=self.operator
        )
        return test_loss
        
    def dump_state(self):
        """Return the state of the model
        
        Returns
        ----------
        state
            State dictionary of the base-learner model
        """
        return self.baselearner.state_dict()
    
    def load_state(self, state):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : dict
            Base-learner parameters
        """
        
        # Load state is only called to load task-model architectures,
        # so call eval mode (because number of classes in task model differs
        # from that of the non-task model
        self.baselearner.eval()
        self.baselearner.load_state_dict(state) 