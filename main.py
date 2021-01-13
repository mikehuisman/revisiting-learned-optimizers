"""
Script to run experiments with a single algorithm of choice.
The design allows for user input and flexibility. 

Command line options are:
------------------------
runs : int, optional
    Number of experiments to perform (using different random seeds)
    (default = 1)
N : int, optional
    Number of classes per task
k : int
    Number of examples in the support sets of tasks
k_test : int
    Number of examples in query sets of meta-validation and meta-test tasks
T : int
    Number of optimization steps to perform on a given task
train_batch_size : int, optional
    Size of minibatches to sample from META-TRAIN tasks (or size of flat minibatches
    when the model requires flat data and batch size > k)
    Default = k (no minibatching, simply use entire set)
test_batch_size : int, optional
    Size of minibatches to sample from META-[VAL/TEST] tasks (or size of flat minibatches
    when the model requires flat data and batch size > k)
    Default = k (no minibatching, simply use entire set)
logfile : str
    File name to write results in (does not have to exist, but the containing dir does)
seed : int, optional
    Random seed to use
cpu : boolean, optional
    Whether to use cpu

Usage:
---------------
python main.py --arg=value --arg2=value2 ...
"""

import argparse
import csv
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

from tqdm import tqdm #Progress bars
from networks import SineNetwork, Conv4, BoostedConv4
from algorithms.metalearner_lstm import LSTMMetaLearner  
from algorithms.train_from_scratch import TrainFromScratch
from algorithms.finetuning import FineTuning
from algorithms.moso import MOSO
from algorithms.turtle import Turtle
from algorithms.maml import MAML
from algorithms.modules.utils import get_init_score_and_operator
from sine_loader import SineLoader
from image_loader import ImageLoader
from misc import BANNER, NAMETAG
from configs import TFS_CONF, FT_CONF, CFT_CONF, LSTM_CONF,\
                    MAML_CONF, MOSO_CONF, TURTLE_CONF

FLAGS = argparse.ArgumentParser()

# Required arguments
FLAGS.add_argument("--problem", choices=["sine", "min", "cub"], required=True,
                   help="Which problem to address?")

FLAGS.add_argument("--k", type=int, required=True,
                   help="Number examples per task set during meta-training."+\
                   "Also the number of examples per class in every support set.")

FLAGS.add_argument("--k_test", type=int, required=True,
                   help="Number examples per class in query set")

FLAGS.add_argument("--model", choices=["tfs", "finetuning", "centroidft", 
                   "lstm", "maml", "moso", "turtle"], required=True,
                   help="Which model to use?")

# Optional arguments
FLAGS.add_argument("--N", type=int, default=None,
                   help="Number of classes (only applicable when doing classification)")   

FLAGS.add_argument("--meta_batch_size", type=int, default=1,
                   help="Number of tasks to compute outer-update")   

FLAGS.add_argument("--val_after", type=int, default=None,
                   help="After how many episodes should we perform meta-validation?")


FLAGS.add_argument("--decouple", type=int, default=None,
                   help="After how many train tasks switch from meta-mode to base-mode?")

FLAGS.add_argument("--lr", type=float, default=None,
                   help="Learning rate for (meta-)optimizer")

FLAGS.add_argument("--cpe", type=int, default=4,
                   help="#Times best weights get reconsidered per episode (only for baselines)")

FLAGS.add_argument("--T", type=int, default=None,
                   help="Number of weight updates per training set")

FLAGS.add_argument("--T_test", type=int, default=None,
                   help="Number of weight updates at test time")

FLAGS.add_argument("--history", choices=["none", "grads", "updates"], default="none",
                   help="Historical information to use (only applicable for TURTLE): none/grads/updates")

FLAGS.add_argument("--beta", type=float, default=None,
                   help="Beta value to use (only applies when model=TURTLE)")

FLAGS.add_argument("--train_batch_size", type=int, default=None,
                   help="Size of minibatches for training "+\
                         "only applies for flat batch models")

FLAGS.add_argument("--test_batch_size", type=int, default=None,
                   help="Size of minibatches for testing (default = None) "+\
                   "only applies for flat-batch models")

FLAGS.add_argument("--activation", type=str, choices=["relu", "tanh", "sigmoid"],
                   default=None, help="Activation function to use for TURTLE/MOSO")

FLAGS.add_argument("--runs", type=int, default=30, 
                   help="Number of runs to perform")

FLAGS.add_argument("--devid", type=int, default=None, 
                   help="CUDA device identifier")

FLAGS.add_argument("--second_order", action="store_true", default=False,
                   help="Use second-order gradient information for TURTLE")

FLAGS.add_argument("--input_type", choices=["raw_grads", "raw_loss_grads", 
                   "proc_grads", "proc_loss_grads", "maml"], default=None, 
                   help="Input type to the network (only for MOSO and TURTLE"+\
                   " choices = raw_grads, raw_loss_grads, proc_grads, proc_loss_grads, maml")

FLAGS.add_argument("--layer_wise", action="store_true", default=False,
                   help="Whether TURTLE should use multiple meta-learner networks: one for every layer in the base-learner")

FLAGS.add_argument("--param_lr", action="store_true", default=False,
                   help="Whether TURTLE should learn a learning rate per parameter")

FLAGS.add_argument("--model_spec", type=str, default=None,
                   help="Store results in file ./results/problem/k<k>test<k_test>/<model_spec>/")

FLAGS.add_argument("--layers", type=str, default=None,
                   help="Neurons per hidden/output layer split by comma (e.g., '10,10,1')")

FLAGS.add_argument("--cross_eval", default=False, action="store_true",
                   help="Evaluate on tasks from different dataset (cub if problem=min, else min)")

FLAGS.add_argument("--seed", type=int, default=1337,
                   help="Random seed to use")

FLAGS.add_argument("--single_run", action="store_true", default=False,
                   help="Whether the script is run independently of others for paralellization. This only affects the storage technique.")

FLAGS.add_argument("--cpu", action="store_true",
                   help="Use CPU instead of GPU")

FLAGS.add_argument("--time_input", action="store_true", default=False,
                   help="Add a timestamp as input to TURTLE")                   

FLAGS.add_argument("--validate", action="store_true", default=False,
                   help="Validate performance on meta-validation tasks")

RESULT_DIR = "./results/"

def create_dir(dirname):
    """
    Create directory <dirname> if not exists
    """
    
    if not os.path.exists(dirname):
        print(f"[*] Creating directory: {dirname}")
        try:
            os.mkdir(dirname)
        except FileExistsError:
            # Dir created by other parallel process so continue
            pass

def print_conf(conf):
    """Print the given configuration
    
    Parameters
    -----------
    conf : dictionary
        Dictionary filled with (argument names, values) 
    """
    
    print(f"[*] Configuration dump:")
    for k in conf.keys():
        print(f"\t{k} : {conf[k]}")

def set_batch_size(conf, args, arg_str):
    value = getattr(args, arg_str)
    # If value for argument provided, set it in configuration
    if not value is None:
        conf[arg_str] = value
    else:
        try:
            # Else, try to fetch it from the configuration
            setattr(args, arg_str, conf[arg_str]) 
            args.train_batch_size = conf["train_batch_size"]
        except:
            # In last case (nothing provided in arguments or config), 
            # set batch size to N*k
            num = args.k
            if not args.N is None:
                num *= args.N
            setattr(args, arg_str, num)
            conf[arg_str] = num             

def overwrite_conf(conf, args, arg_str):
    # If value provided in arguments, overwrite the config with it
    value = getattr(args, arg_str)
    if not value is None:
        conf[arg_str] = value
    else:
        # Try to fetch argument from config, if it isnt there, then the model
        # doesn't need it
        try:
            setattr(args, arg_str, conf[arg_str])
        except:
            return
        
def setup(args):
    """Process arguments and create configurations
        
    Process the parsed arguments in order to create corerct model configurations
    depending on the specified user-input. Load the standard configuration for a 
    given algorithm first, and overwrite with explicitly provided input by the user.

    Parameters
    ----------
    args : cmd arguments
        Set of parsed arguments from command line
    
    Returns
    ----------
    args : cmd arguments
        The processed command-line arguments
    conf : dictionary
        Dictionary defining the meta-learning algorithm and base-learner
    data_loader
        Data loader object, responsible for loading data
    """
    
    args.resdir = RESULT_DIR
    # Ensure that ./results directory exists
    create_dir(args.resdir)
    args.resdir += args.problem + '/'
    # Ensure ./results/<problem> exists
    create_dir(args.resdir)
    if args.N:
        args.resdir += 'N' + str(args.N) + 'k' + str(args.k) + "test" + str(args.k_test) + '/' 
    else:
        args.resdir += 'k' + str(args.k) + "test" + str(args.k_test) + '/' 
    # Ensure ./results/<problem>/k<k>test<k_test> exists
    create_dir(args.resdir)
    if args.model_spec is None:
        args.resdir += args.model + '/'
    else:
        args.resdir += args.model_spec + '/'
    # Ensure ./results/<problem>/k<k>test<k_test>/<model>/ exists
    create_dir(args.resdir)

    # If args.single_run is true, we should store the results in a directory runs
    if args.single_run or args.runs < 30:
        args.resdir += "runs/"
        create_dir(args.resdir)
        args.resdir += f"run{args.seed}-" 
    
    # Mapping from model names to configurations
    mod_to_conf = {
        "tfs": (TrainFromScratch, TFS_CONF),
        "finetuning": (FineTuning, FT_CONF),
        "centroidft": (FineTuning, CFT_CONF), 
        "lstm": (LSTMMetaLearner, LSTM_CONF),
        "maml": (MAML, MAML_CONF),
        "moso": (MOSO, MOSO_CONF),
        "turtle": (Turtle, TURTLE_CONF)
    }

    baselines = {"tfs", "finetuning", "centroidft"}
    
    # Get model constructor and config for the specified algorithm
    model_constr, conf = mod_to_conf[args.model]

    # Set batch sizes
    set_batch_size(conf, args, "train_batch_size")
    set_batch_size(conf, args, "test_batch_size")
        
    # Set values of T, lr, and input type
    overwrite_conf(conf, args, "T")
    overwrite_conf(conf, args, "lr")
    overwrite_conf(conf, args, "input_type")
    overwrite_conf(conf, args, "beta")
    overwrite_conf(conf, args, "meta_batch_size")
    overwrite_conf(conf, args, "time_input")
    
    # Parse the 'layers' argument
    if not args.layers is None:
        try:
            layers = [int(x) for x in args.layers.split(',')]
        except:
            raise ValueError(f"Error while parsing layers argument {args.layers}")
        conf["layers"] = layers
    
    # Make sure argument 'val_after' is specified when 'validate'=True
    if args.validate:
        assert not args.val_after is None,\
                    "Please specify val_after (number of episodes after which to perform validation)"
    
    # If using multi-step maml, perform gradient clipping with -10, +10
    if not conf["T"] is None:
        if conf["T"] > 1 and (args.model=="maml" or args.model=="turtle"):
            conf["grad_clip"] = 10
        elif args.model == "lstm":
            conf["grad_clip"] = 0.25 # it does norm clipping
        else:
            conf["grad_clip"] = None
    
    # If MOSO or TURTLE is selected, set the activation function
    if args.activation:
        act_dict = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(), 
            "sigmoid": nn.Sigmoid()
        }
        conf["act"] = act_dict[args.activation]
    
    # Set the number of reconsiderations of best weights during meta-training episodes,
    # and the device to run the algorithms on 
    conf["cpe"] = args.cpe
    conf["dev"] = args.dev
    conf["second_order"] = args.second_order
    conf["history"] = args.history
    conf["layer_wise"] = args.layer_wise
    conf["param_lr"] = args.param_lr
    conf["decouple"] = args.decouple

    if args.T_test is None:
        conf["T_test"] = conf["T"]
    else:
        conf["T_test"] = args.T_test
    

    assert not (args.input_type == "maml" and args.history != "none"), "input type 'maml' and history != none are not compatible"
    assert not (conf["T"] == 1 and args.history != "none"), "Historical information cannot be used when T == 1" 

    # Different data set loader to test domain shift robustness
    cross_loader = None
    
    # Pick appropriate base-learner model for the chosen problem [sine/image]
    # and create corresponding data loader obejct
    if args.problem == "sine":
        data_loader = SineLoader(k=args.k, k_test=args.k_test, seed=args.seed)
        conf["baselearner_fn"] = SineNetwork
        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev}
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
            "reset_ptr": True,
        }
    else:
        assert not args.N is None, "Please provide the number of classes N per set"
        
        if args.k >= 5:
            train_iters = 40000
        else:
            train_iters = 60000
        eval_iters = 600

        if "min" in args.problem:
            data_loader = ImageLoader(N=args.N, k=args.k, k_test=args.k_test, 
                                    num_train_tasks=train_iters, num_val_tasks=eval_iters,
                                    img_size=(84,84), path="./data/min/", seed=args.seed)
            if args.cross_eval:
                cross_loader = ImageLoader(N=args.N, k=args.k, k_test=args.k_test, 
                                           num_train_tasks=train_iters, num_val_tasks=eval_iters, 
                                           img_size=(84,84), path="./data/cub/", cross=True, seed=args.seed)
        else:
            if args.cross_eval:
                cross_loader = ImageLoader(N=args.N, k=args.k, k_test=args.k_test, 
                                           num_train_tasks=train_iters, num_val_tasks=eval_iters, 
                                           img_size=(84,84), path="./data/min/", cross=True, seed=args.seed)
                
            data_loader = ImageLoader(N=args.N, k=args.k, k_test=args.k_test, 
                                      num_train_tasks=train_iters, num_val_tasks=eval_iters,
                                      img_size=(84,84), path="./data/cub/", seed=args.seed)
        
        # Image problem
        if args.model == "centroidft":
            conf["baselearner_fn"] = BoostedConv4
        else:    
            conf["baselearner_fn"] = Conv4
        
        if args.model in baselines:
            if not args.model == "tfs":
                train_classes = data_loader.total_classes(mode="train")
            else:
                train_classes = args.N # TFS does not train, so this enforces the model to have the correct output dim. directly
        else:
            train_classes = args.N
            
        conf["baselearner_args"] = {
            "train_classes": train_classes,
            "eval_classes": args.N, 
            "criterion": nn.CrossEntropyLoss(),
            "dev":args.dev
        }
        
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only used for baselines
            "reset_ptr": True
        }

    # Print the configuration for confirmation
    print_conf(conf)
    return args, conf, data_loader, cross_loader, model_constr

def validate(model, data_loader, best_score, best_state, conf, args):
    """Perform meta-validation
        
    Create meta-validation data generator obejct, and perform meta-validation.
    Update the best_loss and best_state if the current loss is lower than the
    previous best one. 

    Parameters
    ----------
    model : Algorithm
        The chosen meta-learning model
    data_loader : DataLoader
        Data container which can produce a data generator
    best_score : float
        Best validation performance obtained so far
    best_state : nn.StateDict
        State of the meta-learner which gave rise to best_loss
    args : cmd arguments
        Set of parsed arguments from command line
    
    Returns
    ----------
    best_loss
        Best obtained loss value so far during meta-validation
    best_state
        Best state of the meta-learner so far
    score 
        Performance score on this validation run
    """
    
    print("[*] Validating performance...")
    scores = []
    val_loader = data_loader.generator(episodic=True, mode="val", **conf["generator_args"])
    for epoch in val_loader:
        train_x, train_y ,test_x, test_y = epoch
        score = model.evaluate(train_x = train_x, 
                               train_y = train_y, 
                               test_x = test_x, 
                               test_y = test_y)
        scores.append(score)
    
    score = np.mean(scores)
    # Compute min/max (using model.operator) of new score and best score 
    tmp_score = model.operator(score, best_score)
    # There was an improvement, so store info
    if tmp_score != best_score and not math.isnan(tmp_score):
        best_score = score
        best_state = model.dump_state()
    print("validation loss:", score)
    return best_score, best_state, score
        
def body(args, conf, data_loader, cross_loader, model_constr):
    """Create and apply the meta-learning algorithm to the chosen data
    
    Backbone of all experiments. Responsible for:
    1. Creating the user-specified model
    2. Performing meta-training
    3. Performing meta-validation
    4. Performing meta-testing
    5. Logging and writing results to output channels
    
    Parameters
    -----------
    args : arguments
        Parsed command-line arguments
    conf : dictionary
        Configuration dictionary with all model arguments required for construction
    data_loader : DataLoader
        Data loder object which acts as access point to the problem data
    model_const : constructor fn
        Constructor function for the meta-learning algorithm to use
    
    """
        
    # Write learning curve to file "curves<val_after>.csv"    
    curvesfile = args.resdir+"curves"+str(args.val_after)+".csv"
    
    overall_best_score = get_init_score_and_operator(conf["baselearner_args"]["criterion"])[0]
    overall_best_state = None
    print("overall best score:", overall_best_score)
    
    seeds = [random.randint(0, 100000) for _ in range(args.runs)]
    print("Actual seed:", seeds)

    for run in range(args.runs):
        stime = time.time()
        print("\n\n"+"-"*40)
        print(f"[*] Starting run {run}")
        # Set torch seed to ensure same base-learner initialization across techniques
        torch.manual_seed(seeds[run])
        model = model_constr(**conf)

        if model.operator == max:
            logstr = "accuracy"
        else:
            logstr = "loss"
        
        vtime = time.time()
        # Start with validation to ensure non-trainable model get 
        # validated at least once
        if args.validate:
            best_score, best_state = model.init_score, None
            best_score, best_state, score = validate(model, data_loader, 
                                                     best_score, best_state, 
                                                     conf, args)
            print(f"[*] Done validating, cost: {time.time()-vtime} seconds")
            # Stores all validation performances over time (learning curve) 
            learning_curve = [score]
            
        
        if model.trainable:
            dcounter = [1,0] if conf["decouple"] else [0]

            print('\n[*] Training...')
            ttime = time.time()
            for el in dcounter:
                train_generator = data_loader.generator(episodic=model.episodic, mode="train", 
                                                        **conf["generator_args"])

                for eid, epoch in enumerate(train_generator):
                    #task_time = time.time()
                    # Unpack the episode. If the model is non-episodic in nature, test_x and 
                    # test_y will be None
                    train_x, train_y, test_x, test_y = epoch
                    # Perform update using selected batch
                    model.train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
                    #print(time.time() - task_time, "seconds")
                    #task_time = time.time()


                    # Perform meta-validation
                    if args.validate and (eid + 1) % args.val_after == 0 and el != 1:
                        print(f"{time.time() - ttime} seconds for training")
                        ttime = time.time()
                        vtime = time.time()
                        best_score, best_state, score = validate(model, data_loader, 
                                                                best_score, best_state, 
                                                                conf, args)
                        print(f"[*] Done validating, cost: {time.time()-vtime} seconds")
                        # Store validation performance for the learning curve
                        # note that score is more informative than best_score 
                        learning_curve.append(score)
                
        if args.validate:
            # Load best found state during meta-validation
            model.load_state(best_state)
        
        test_generator = data_loader.generator(episodic=True, mode="test", **conf["generator_args"])
        generators = [test_generator]
        filenames = [args.resdir+"scores.csv"]
        
        if args.cross_eval:
            cross_generator = cross_loader.generator(episodic=True, mode="test", **conf["generator_args"])
            crossfile = args.resdir+"cross_scores.csv" 
            generators.append(cross_generator)
            filenames.append(crossfile)
        
        for idx, (eval_gen, filename) in enumerate(zip(generators, filenames)):
            test_scores = []
            print('\n[*] Evaluating test performance...')
            
            for epoch in eval_gen:
                train_x, train_y, test_x, test_y  = epoch 

                score = model.evaluate(
                        train_x = train_x, 
                        train_y = train_y, 
                        test_x = test_x, 
                        test_y = test_y, 
                )
                test_scores.append(score)

            # Log the test scorees to logfile
            if not os.path.exists(filename) or run == 0:
                with open(filename, "w+", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["run",f"mean_{logstr}",f"median_{logstr}"])
            r, mean, median = str(run), str(np.mean(test_scores)),\
                            str(np.median(test_scores))
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([r, mean, median])
            print(f"Run {run} done, mean {logstr}: {mean}, median {logstr}: {median}")
            print(f"Time used: {time.time() - stime}")
            print("-"*40)
        
        if args.validate:
            print(learning_curve)
            # Determine writing mode depending on whether learning curve file already exists
            # and the current run
            if not os.path.exists(curvesfile) or run == 0:
                open_mode = "w+"
            else:
                open_mode = "a"
            # Write learning curve to file
            with open(curvesfile, open_mode, newline="") as f:
                writer = csv.writer(f)
                writer.writerow([str(score) for score in learning_curve])
            
            # Check if the best score is better than the overall best score
            # if so, update best score and state across runs. 
            # It is better if tmp_best != best
            tmp_best_score = model.operator(best_score, overall_best_score)
            if tmp_best_score != overall_best_score and not math.isnan(tmp_best_score):
                print(f"[*] Updated best model configuration across runs")
                overall_best_score = best_score
                overall_best_state = best_state
    
    # At the end of all runs, write the best found configuration to file
    if args.validate:            
        save_path = args.resdir+"model.pkl"
        print(f"[*] Writing best model state to {save_path}")
        model.load_state(overall_best_state)
        model.store_file(save_path)

if __name__ == "__main__":
    # Parse command line arguments
    args, unparsed = FLAGS.parse_known_args()

    # If there is still some unparsed argument, raise error
    if len(unparsed) != 0:
        raise ValueError(f"Argument {unparsed} not recognized")
    
    # Set device to cpu if --cpu was specified
    if args.cpu:
        args.dev="cpu"
    
    # If cpu argument wasn't given, check access to CUDA GPU
    # defualt device is cuda:1, if that raises an exception
    # cuda:0 is used
    if not args.cpu:
        print("Current device:", torch.cuda.current_device())
        print("Available devices:", torch.cuda.device_count())
        if not args.devid is None:
            torch.cuda.set_device(args.devid)
            args.dev = f"cuda:{args.devid}"
            print("Using cuda device: ", args.dev)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU unavailable.")
            try:
                torch.cuda.set_device(1)
                args.dev="cuda:1"
            except:
                torch.cuda.set_device(0)
                args.dev="cuda:0"

    # Let there be reproducibility!
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Chosen seed:", args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   
    # Let there be recognizability!
    print(BANNER)
    print(NAMETAG)

    # Let there be structure!
    pargs, conf, data_loader, cross_loader, model_constr = setup(args)
    
    # Let there be beauty!
    body(pargs, conf, data_loader, cross_loader, model_constr)