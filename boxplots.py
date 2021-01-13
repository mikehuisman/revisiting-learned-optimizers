from analyzer import RESDIR, enumerator, print_options, cprint, oprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

to_plot = {
    "FO-maml-1step": ("MAML", "First-order"), 
    "SO-maml-1step": ("MAML", "Second-order"),
    "turtleFO-turtle1-raw_grads-relu": ("0", "First-order"),
    "turtleSO-turtle1-raw_grads":  ("0", "Second-order"),
    "turtleFO-turtle20-1-raw_grads-relu":  ("1", "First-order"),
    "turtleSO-turtle20-1-raw_grads-relu":  ("1", "Second-order"),
    "turtleFO-turtle20-20-1-raw_grads-relu":  ("2", "First-order"),
    "turtleSO-turtle20-20-1-raw_grads-relu": ("2", "Second-order"),
    "turtleFO-turtle20-20-20-1-raw_grads-relu": ("3", "First-order"),
    "turtleSO-turtle20-20-20-1-raw_grads-relu": ("3", "Second-order"),
    "turtleFO-turtle20-20-20-20-1-raw_grads-relu": ("4", "First-order"),
    "turtleSO-turtle20-20-20-20-1-raw_grads-relu": ("4", "Second-order"),
    "turtleFO-turtle-5HL-raw_grads-relu": ("5", "First-order"),
    "turtleSO-turtle20-20-20-20-20-1-raw_grads-relu": ("5", "Second-order"),
    "turtleFO-turtle-6HL-raw_grads-relu": ("6", "First-order"),
    "turtleSO-turtle-6HL-raw_grads-relu": ("6", "Second-order"),
    "turtleFO-turtle-7HL-raw_grads-relu": ("7", "First-order"),
    "turtleSO-turtle-7HL-raw_grads-relu": ("7", "Second-order")
}

PROBLEM = "k5test50"

cdir = RESDIR + "sine/"
file = "curves" # Learning curve files should have this in their filename
markers = ['o', 'v', '^', 's', 'p', 'P', 'D']
mc = 0 # Index for markers

# Generate options for algorithms to include in the plot
options = [x for x in enumerator(cdir, file)]
rd_options = [(x[0],x[1], x[2].split('/')[-1]) for x in options]
indices = []
count = 0
for alg, problem, path in rd_options:
    if alg in to_plot and problem==PROBLEM:
        indices.append(count)
        print("HERE:",path)
    count += 1
        
subset = np.array(options)[np.array(indices)]


fdf = pd.DataFrame(columns=["Algorithm", "Gradients", "MSE"])

sindex = 0 # Index of first episode to plot
upper_bounds = []
lower_bounds = []
lcurves = []
axes = []
labels = []
# All learning curves of interest
for alg, problem, path in subset:
    # Either the user inputs the label he/she wishes or we use the 
    # algorithm filename by default
    label = to_plot[alg][0]

    # Construct learning curve from file and infer error bounds 
    df = pd.read_csv(path, header=None)
    best_perfs = df.min(axis=1) # List of all best validation performances per run
    alg_name = [to_plot[alg][0] for _ in range(len(best_perfs))]
    order = [to_plot[alg][1] for _ in range(len(best_perfs))]
    
    
    perf_col = list(fdf["MSE"]) + list(best_perfs)
    alg_col = list(fdf["Algorithm"]) + alg_name
    order_col = list(fdf["Gradients"]) + order
    
    fdf = pd.DataFrame(columns=["Algorithm"])
    
    fdf["MSE"] = pd.Series(perf_col)
    fdf["Algorithm"] = pd.Series(alg_col)
    fdf["Gradients"] = pd.Series(order_col)
    
fdf["Algorithm"] = pd.Categorical(fdf['Algorithm'], ["MAML", "0", "1", "2", "3", 
                                  "4", "5", "6", "7"])

print(fdf)
print(set(fdf["Algorithm"]))

plt.figure(figsize=(16,14))
#plt.title("T = 1 step", fontsize=50)
ax = sns.boxplot(x="Algorithm", y="MSE", hue="Gradients", data=fdf.sort_values(by=["Algorithm","Gradients"]), palette="Set1", showfliers=False)
ax.set_xlabel("Algorithm",fontsize=48)
ax.set_ylabel("MSE loss", fontsize=48)
for l in ax.lines:
    l.set_linewidth(5)
plt.legend(fontsize=45)
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.savefig("boxplot-adj2.png")