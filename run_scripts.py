import subprocess

with open("seeds.txt", "r") as f:
    seeds = [int(x.strip()) for x in f.readlines()]

    
for i in range(0, len(seeds), 3):
    # Run 3 tasks at most in parallel
    subset = seeds[i: i+3]
    print(subset, "running")
    
    processes = []
    for seed in subset:
        command = f"python -u main.py --problem sine --k_test 50 --k 5 --model turtle --validate --val_after 2500 --lr 0.001 --cpu --second_order --model_spec turtleSO20-20-1-5step-raw_grads --layers 20,20,1 --input_type raw_grads --T 5 --runs 1 --single_run --seed {seed} > ./logfiles/SOturtle20-20-1-5step/run{seed}.txt"
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(proc)

    # Wait for processes to end
    for p in processes:
        p.wait()
    