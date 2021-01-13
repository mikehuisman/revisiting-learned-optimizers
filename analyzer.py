import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from os.path import join 

"""
Script for the analysis of obtained results

...

Methods
-------
cprint(string)
    Custom print function which starts with '[*]'
oprint(string)
    Custom print function that starts with an indent
print_options(options)
    Prints the possible options
enum_dir(dirstr)
    Enumerate the directory given by dirstr
parse_input(inp,options)
    Parses the user-input inp and checks whether it is a valid option
select_problem(cdir)
    Select which problem to analyze (sine/miniImageNet/CUB/...)
enumerator(cdir,file)
    Enumerate all result files in the current directory (cdir)
table(cdir)
    Interactively loads performance data and creates a latex table
plot(cdir)
    Interactively loads learning curves and plots them
select_mode()
    Asks user whether to create a table or plot
"""

PREF = "[*]"
OPT = "   "
RESDIR = "./results/"

def cprint(string):
    """Prepend [*] to the string to be printed
    
    Parameters
    ----------
    string : str
        String to be printed
    """
    
    print(f"{PREF} {string}")

    
def oprint(string):
    """Prepend an indent to the string to be printed
    
    Parameters
    ----------
    string : str
        String to be printed
    """
    
    print(f"{OPT} {string}")
    
def print_options(options):
    """Prints out options
    
    Print out the options in the format (@index - @optionname)
    
    Parameters
    ----------
    options : list
        List of option strings
    """
    
    cprint("Options:")
    for i, option in enumerate(options):
        oprint(f"{i} - {option}")
    
def enum_dir(dirstr):
    """Enumerate the given directory
    
    Construct a list of all full-path contents of the given directory.
    
    Parameters
    ----------
    dirstr : str
        Directory to enumerate
    
    Returns
    ----------
    options
        List of full-path specifications of the contents of the given directory
    """
    
    options = os.listdir(dirstr)
    print_options(options)
    options = [join(dirstr, option) for option in options]
    return options
    
    
def parse_input(inp, options):
    """Parses user-provided input
    
    Check if the given input can be seen as an index of the options list.
    
    Parameters
    ----------
    inp : str
        User input
    options : list
        List of option strings
    
    Returns
    ----------
    parsed_choice
        The index corresponding to the input if index is valid for the options list
        else None
    """
    
    try:
        parsed_choice = int(inp)
        if parsed_choice >= 0 and parsed_choice < len(options):
            return parsed_choice
    except:
        return None
    
    
def select_problem(cdir):
    """Asks user which problem to address
    
    Enumerate the directory and give all contents as potential problems 
    to tackle. We assume these contents are indeed problem folders (e.g., sine)
    
    Parameters
    ----------
    cdir : str
        Directory where the problems are listed
    
    Returns
    ----------
    option
        The chosen problem to work on
    """
    
    cprint("Select a problem.")
    options = enum_dir(cdir)    
    choice = parse_input(input("Your choice:"), options)
    # If parsing failed, return to select problem
    if choice is None:
        oprint(" ---------- Invalid input detected")
        return select_problem(cdir)
    return options[choice]

def process_runs(path):
    """Process the results from parallel runs
    
    Transform all contents of the runs directory path/runs/ into 2 files
    scores.csv and curves2500.csv
    
    Parameters
    ----------
    path : str
        Path specifier
    """

    rdir = join(path, "runs")
    if os.path.exists(rdir):
        cprint(f"Processing /runs/ folder in {path}")
        files = os.listdir(rdir)
        sfiles = [join(rdir, x) for x in files if "score" in x]
        cfiles = [join(rdir, x) for x in files if "curve" in x]
        if len(cfiles) == 0:
            return
        val_after = int(cfiles[-1].split("/")[-1].split(".csv")[0].split("curves")[-1])
        clines = []
        slines = []
        sheader = None

        # Read all score files and append lines to slines
        for file in sfiles:
            with open(file, "r") as f:
                lines = f.readlines()
                content = lines[1:] # remove :6 when done with 
                sheader = lines[0]
            for l in content:
                slines.append(l)
        # Idem for curve files
        for file in cfiles:
            with open(file, "r") as f:
                content = f.readlines()
            for l in content:
                clines.append(l)

        # # Read all score files and append lines to slines
        # for file in sfiles:
        #     with open(file, "r") as f:
        #         lines = f.readlines()
        #         content = lines[-1]
        #         sheader = lines[0]
        #     slines.append(content)
        # # Idem for curve files
        # for file in cfiles:
        #     with open(file, "r") as f:
        #         content = f.readlines()[0]
        #     clines.append(content)

        curve_file = join(path, f"curves{val_after}.csv")
        scores_file = join(path, "scores.csv")
        slines = [sheader] + slines

        with open(curve_file, "w+") as f:
            f.writelines(clines)

        with open(scores_file, "w+") as f:
            f.writelines(slines)

def enumerator(cdir, file):
    """Enumerate all result files in directory cdir
    
    Traverse the current directory to find file names that contain @file.
    Generate all of them.
    
    Parameters
    ----------
    cdir : str
        Directory to start the traversal process
    file : str
        String which should be present in files of interest
    
    Returns
    ----------
    Generator
        Generator object that yields tuples (alg, problem, filepath)
        which are the algorithm name, problem name, and filepath to the files of interest
    """
    
    subproblems = os.listdir(cdir)
    paths = {x:join(cdir, x) for x in subproblems}
    
    for problem in paths.keys():
        # Path to ptoblem folder e.g. k5test50
        path = paths[problem]
        for alg in os.listdir(path):
            alg_path = join(path, alg)
            #process_runs(alg_path)
            files = [join(alg_path, x) for x in os.listdir(alg_path) if file in x]            
            for filepath in files:
                yield alg, problem, filepath


def process_curves(problem, alg, path, results, minimize=True):
    """Processes a single curves*.csv file
    
    Read the csv from the given path, and aggregate the results
    from the mean and median performance scores. Aggregates are 
    added to the results dictionary

    Params
    ----------
    problem : str
        Problem specifier (e.g., k5test50)
    alg : str
        Algorithm specifier
    path : str
        Full path to the curves*.csv file
    results : dict
        Dictionary with results [algorithm] -> [problem+(mean/median)] 
        -> aggregated result
    """

    print(path)
    df = pd.read_csv(path, header=None)
    # Get best validation score per run
    print(path)
    if minimize:
        best_scores = np.array(df.min(axis=1))
    else:
        best_scores = np.array(df.max(axis=1))

    count = len(best_scores)
    std = np.std(best_scores)
    eb = 1.96*std/np.sqrt(count)
    mean = np.mean(best_scores)
    median = np.median(best_scores)
    mean_display_string = f"{mean:.2f} \\pm {eb:.2f}"
    median_display_string = f"{median:.2f} \\pm {eb:.2f}"
    results[alg][f"{problem}mean"] = mean_display_string
    results[alg][f"{problem}median"] = median_display_string
    print(f"{problem}, {alg} : Entries found: {count}")


def process_scores(problem, alg, path, results, **kwargs):
    """Processes a single scores.csv file
    
    Read the csv from the given path, and aggregate the results
    from the mean and median performance scores. Aggregates are 
    added to the results dictionary

    Params
    ----------
    problem : str
        Problem specifier (e.g., k5test50)
    alg : str
        Algorithm specifier
    path : str
        Full path to the scores.csv file
    results : dict
        Dictionary with results [algorithm] -> [problem+(mean/median)] 
        -> aggregated result
    """

    # Read CSV file, aggregate mean and median scores
    df = pd.read_csv(path, index_col=0)[:]
    print(f"{problem}, {alg} : Entries found: {len(df)}")
    #print(df.columns)
    for col in df.columns: #["mean_loss", "median_loss"]
        # Ignore outliers

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1 
        # outliers are < q1 - 1.5*iqr or > q3 + 1.5*iqr
        sub = (df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)

        mean, count, std = df[col][sub].agg(["mean", "count", "std"])
        # Error bound for 95% confidence-interval
        eb = 1.96*std/np.sqrt(count)
        display_string = f"{mean:.2f} \\pm {eb:.2f}"
        results[alg][f"{problem}{col.split('_')[0]}"] = display_string

def table(cdir):
    """Creates a table of performance files of interest
    
    Traverse the current directory to find "scores.csv" files.
    Aggregate these result files, compute confidence intervals,
    and write the results to a table which is also given as output
    in latex mode.
    
    
    Parameters
    ----------
    cdir : str
        Directory of the problem (e.g., sine)
    """
    
    val = input("Table of validation results (y/default=n) or cub?")
    # The files that we are looking for
    if val == 'y':
        file = "curves"
        fn = process_curves
        minmax = input("Minimize the score (y/n=default -> maximize)")
        minimize = minmax == "y"
    elif val == "cub":
        file = "cross_scores.csv"
        fn = process_scores
        minimize = None
    else:
        file = "scores.csv"
        fn = process_scores
        minimize = None
    
    results = dict()
    
    for alg, problem, path in enumerator(cdir, file):
        if not alg in results:
            results[alg] = dict()

        fn(problem, alg, path, results, minimize=minimize)
        
    table = pd.DataFrame.from_dict(results, orient="index")
    print(table)
    filename = input("Store table as: ")
    with open(filename, "w+") as f:
        f.write(table.to_latex(escape=False))


def plot_curves(**args):
    """Plot all given curves

    Parameters
    ----------
    **args : dict
        Keyword arguments required for plot_curves with keys:
        "axes", "lcurves", "lbounds": lower_bounds, "ubounds", "labels", "markers", "findex"
    """

    findex = args["findex"]
    for x, lc, lb, ub, label, marker in zip(args["axes"], args["lcurves"], 
                                            args["lbounds"], args["ubounds"], 
                                            args["labels"], args["markers"]):

        plt.plot(x[findex:], lc[findex:], label=label, marker=marker, markersize=5) 
        plt.fill_between(x[findex:], lb[findex:], ub[findex:], alpha=0.3)
    plt.legend()


def set_title():
    """Set title of plot
    """

    title = input("Title: ")
    plt.title(title, fontsize=16)

def set_xaxis():
    """Set x-axis label of plot
    """
    
    xaxis = input("X-axis label: ")
    plt.xlabel(xaxis, fontsize=14)

def set_yaxis():
    """Set y-axis label of plot
    """
    
    yaxis = input("Y-axis label: ")
    plt.ylabel(yaxis, fontsize=14)

def start_index(**args):
    """Gets the index of the first episode that should be plotted

    Returns
    ----------
    index
        First episode index that should be plotted
    """
    
    index = int(input("First episode to plot: "))
    args["findex"] = index
    plt.clf()
    plot_curves(**args)

def save_figure():
    """Save the figure
    """

    filename = input("Save figure as: ")
    plt.savefig(filename)

def quit():
    """Quit the program
    """
    print("Halting the program.")
    import sys; sys.exit()

def interact(**args):
    """Start interactive plot session

    Creates an interactive plot where user input is displayed in the figure in real-time

    Parameters
    ----------
    **args : dict
        Keyword arguments required for plot_curves with keys:
        "axes", "lcurves", "lbounds": lower_bounds, "ubounds", "labels", "markers", "findex"
    """

    options = ["Set title", "Set x-label", "Set y-label", "Set start index", "Save figure", "Quit"]
    functions = [set_title, set_xaxis, set_yaxis, start_index, save_figure, quit]
    arguments = [{}, {}, {}, args, {}, {}]

    plt.show(block=False)
    while True:
        "Please select an option from the ones below:"
        print_options(options)
        option = parse_input(input("Selected option: "), options)
        # Invalid option so try again
        if option is None:
            continue

        # Else, execute the option
        functions[option](**arguments[option])

def plot(cdir):
    """Interactively creates a plot of learning curves of interest
    
    Enumerate all learning curves that exist in the traversed current 
    directory. Ask the user which ones he/she would like to see in a plot
    and do precisely that. We also compute confidence intervals to generate
    error bounds for every included curve.
    
    Parameters
    ----------
    cdir : str
        Directory of the problem (e.g., sine)
    """
    
    file = "curves" # Learning curve files should have this in their filename
    markers = ['o', 'v', '^', 's', 'p', 'P', 'D']
    mc = 0 # Index for markers
    
    # Generate options for algorithms to include in the plot
    options = [x for x in enumerator(cdir, file)]
    rd_options = [(x[0],x[1], x[2].split('/')[-1]) for x in options]
    print_options(rd_options)
    
    # Ask user to make a selection of provided options to draw in a plot
    selection = input("Selection (e.g. '0,5,10'): ")
    try:
        indices = [int(x) for x in selection.split(',')]
    except:
        cprint("Error parsing selection. Retry:")
        return plot(cdir)
    
    # Read all selected learning curves
    options = np.array(options)
    subset = options[indices]
    
    sindex = 0 # Index of first episode to plot
    upper_bounds = []
    lower_bounds = []
    lcurves = []
    axes = []
    labels = []
    # All learning curves of interest
    for alg, problem, path in subset:
        # Provide labels to be used in the plot
        cprint(f"{alg}, {problem}, {path.split('/')[-1]}")
        # Either the user inputs the label he/she wishes or we use the 
        # algorithm filename by default
        inp = input(f"Use label (default= {alg}): ")
        if inp.strip() == '':
            label = alg
        else:
            label = inp
        
        # Construct learning curve from file and infer error bounds 
        df = pd.read_csv(path, header=None)
        lc = np.array(df.mean(axis=0)) # Average over all learning curves
        count = len(df)
        std = np.array(df.std(axis=0))
        oprint(f"Entries found: {count}")
        error = 1.96*std/np.sqrt(count)
        # Upper and lower bounds using 95% CI error bounds 
        ub = lc + error
        lb = lc - error
        # Interval at which validation took place
        interval = int(path.split('/')[-1].split(file)[1].split('.')[0])
        x = [i*interval for i in range(len(lc))]

        lcurves.append(lc)
        lower_bounds.append(lb)
        upper_bounds.append(ub)
        labels.append(label)
        axes.append(x)

    # Create figure
    plt.figure()
    # Make it interactive if wanted
    interactive = input("Enable interactive mode (y/default=n): ")
    args = {"axes":axes, "lcurves": lcurves, "lbounds": lower_bounds, 
                "ubounds":upper_bounds, "labels":labels, "markers":markers, "findex":0}

    if interactive == "y":
        plt.ion()
        interact(**args)
    else:
        # Ask for plot meta-data 
        ptitle = input("plot title: ")
        xaxis = input("x-axis title (default: Episode): ")
        if xaxis == '':
            xaxis = "Episode"
        yaxis = input("y-axis title: ")
        sepisode = input("Starting episode (default=0): ")
        try:
            if sepisode != "":
                args["findex"] = int(sepisode)
        except:
            oprint("Error parsing input.")

        plot_curves(**args)
        savename = input("Save plot as: ")
        plt.title(ptitle, fontsize=14)
        plt.xlabel(xaxis, fontsize=12)
        plt.ylabel(yaxis, fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend()
        plt.savefig(savename)

    
def select_mode():
    """Selects whether to create a plot or table
    
    Returns
    ----------
    function : fn
        Function that does the wanted thing (e.g., create plot)
    """
    
    cprint("What do you want to do?")
    options = ["Create table", "Create plot"]
    functions = [table, plot]
    print_options(options)
    choice = parse_input(input("Your choice:"), options)
    return functions[choice]
    
    
if __name__ == "__main__":
    cprint("You have summoned a RAT (Result Analysis Tool).")
    
    CDIR = RESDIR
    CDIR = select_problem(CDIR)
    
    fn = select_mode()
    fn(CDIR)