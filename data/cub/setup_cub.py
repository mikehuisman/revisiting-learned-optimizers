import gdown
import tarfile
import os
import shutil
import numpy as np

from os.path import join

PREF = " "*5
URL = "https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45&export=download"
OUTFILE = "CUB_200_2011.tgz"
UNTAR = "CUB_200_2011"

def setup_cub():
    """
    Downloads all required files for the CUB dataset, and performs
    all steps necessary to use it. 
    """
    
    print(f"[*] Checking whether {UNTAR} exists...")
    if os.path.exists(UNTAR):
        print(f"{PREF}- File found. ")
        untar = False
        jmp = True
    else:
        print(f"{PREF}- Failed to find file")
        jmp = False

    if not jmp:
        untar = True
        print(f"[*] Checking whether {OUTFILE} exists...")
        if os.path.exists(OUTFILE):
            print(f"{PREF}- File found. ")
        else:
            print(f"{PREF}- Failed to find file")
            print(f"{PREF}- Trying to download now...")
            gdown.download(URL, OUTFILE, quiet=False)
            print("[*] Download successful.")

    if untar:
        print("[*] Unpacking tarball...")
        tar = tarfile.open(OUTFILE)
        tar.extractall()
        tar.close()
        print(f"{PREF}- Succes.")

        print(f"[*] Removing {OUTFILE}")
        os.remove(OUTFILE)
        print(f"{PREF}- Successfully removed tarball")
        print(f"[*] Removing ./attributes.txt")
        os.remove("./attributes.txt")
        print(f"{PREF}- Ok")

    print("[*] Checking whether ./images exists")
    if os.path.exists("./images"):
        print(f"{PREF}- Found.")
    else:
        print("[*] Extracting image folder...")
        shutil.copytree(UNTAR+"/images/", "./images/")
        print(f"{PREF}- Removing {UNTAR}")
        shutil.rmtree(UNTAR)
        print(f"{PREF}- Done")

    # Create train/val/test splits of labels
    # path to images
    PATH = "./images/"

    # class directories 
    class_dirs = os.listdir(PATH)
    num_classes = len(class_dirs)

    # compute bounds to slice the classes into train/val/test splits
    # using a ratio of 70/15/15
    train_size = int(num_classes*0.7)
    val_size = test_size = int(num_classes*0.15)

    # Create splits 
    np.random.seed(1337)
    np.random.shuffle(class_dirs)

    train = class_dirs[:train_size] 
    val = class_dirs[train_size:train_size+val_size]
    test = class_dirs[train_size+val_size:]

    splits = {"train":train,
              "val": val,
              "test": test}

    # Create hierarchy [split] -> {classes] -> [examples with that class]
    for split in splits.keys():
        if not os.path.exists(split):
            print("[*] Creating directory:", split)
            os.mkdir(split)
            print(f"{PREF}- Done")

        print(f"[*] Processing {split} classes")
        # Iterate over all classes of a specific split (train/val/test)
        for class_dir in splits[split]:
            print(f"{PREF}- Class: {class_dir}")
            dirname = join(PATH, class_dir) #./images/classname
            new_path = join(split, class_dir) #./split/classname
            if not os.path.exists(new_path):
                os.mkdir(new_path)

            for file in os.listdir(dirname):
                shutil.copyfile(join(dirname, file),
                                join(new_path, file))

    print("[*] Successfully setup CUB!")