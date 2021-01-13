import os
import csv
import shutil
import sys
import zipfile
import urllib
import gdown

"""
Assumption: Directory ./images/ exists, and contains images required for miniImageNet.
"""

# Print prefix
PREF = " "*5
# Path to the folder containing the images
PATH = "./images/"
SPLITS = ["train", "val", "test"]
SPLITDIR = "./splits/"

IMAGES_URL = "https://drive.google.com/uc?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE&export=download" 
OUTFILE = "images.zip"
UNZIPPED = "./images/"

BASE_URL = "https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/"

def setup_min():
    """
    Downloads all required files for the miniImageNet dataset, and performs
    all steps necessary to use it. 
    """

    # Check if ./images already exists
    print(f"[*] Checking if {UNZIPPED} directory exists")
    if os.path.exists(UNZIPPED):
        print(f"{PREF}- Directory found!")
        # Jump to SPLIT directory checking
        jmp = True
        unzip = False
    else:
        print(f"{PREF} Failed to find directory")
        jmp = False
        unzip = True


    if not jmp:
        # Check if ./images.zip exists
        print(f"[*] Checking if {OUTFILE} exists")
        if not os.path.exists(OUTFILE):
            print(f"{PREF}- Failed to find file. Trying to download now...")
            try:
                gdown.download(IMAGES_URL, OUTFILE, quiet=False) 
                print(f"{PREF}- Successfully downloaded images.")
            except:
                print(f"{PREF}- Download failed. Please try again")
                sys.exit(-1)

    if unzip:
        print(f"{PREF}- Attempting to unzip...") 
        # Here the ZIP file exists, but the images dir does not --> unzip
        with zipfile.ZipFile(OUTFILE, 'r') as f:
            f.extractall()
        print(f"{PREF}- Extraction successful!") 
        print(f"{PREF}- Removing {OUTFILE}...") 
        os.remove(OUTFILE)
        print(f"{PREF}- All set.")


    # Dictionary mapping class identifier -> [list of examples with that class]
    images = dict()
    for image in os.listdir(PATH):
        if not ".jpg" in image:
            continue
        # Class identifier consists of the first 12 characters of the filenames
        class_id = image[:12]
        images[image] = class_id

    print(f"[*] Checking whether directory {SPLITDIR} exists")
    if not os.path.exists(SPLITDIR):
        print(f"{PREF}- Directory not found. Creating now...")
        os.mkdir(SPLITDIR)
        print(f"{PREF}- Succes.")
    else:
        print(f"{PREF}- Found.")

    # Download the train/validation/test splits if they dont exist
    for split in SPLITS:
        splitfile = SPLITDIR + split + ".csv"

        print(f"[*] Attempting to read {splitfile}")
        if not os.path.exists(splitfile):
            print(f"{PREF}- Failed to find file. Trying to download now...")
            urllib.request.urlretrieve(f"{BASE_URL}{split}.csv", splitfile)
            print(f"{PREF}- Download successful.")

        if not os.path.exists(split):
            print(f"[*] Creating directory {split}")
            os.mkdir(split)

        print(f"[*] Processing {split} files")
        # Read the split file containing the filenames to be put in that specific split
        # and copy the files from ./images/ to the corresponding split folder ./split/
        with open(splitfile, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                imagefile, label = line
                assert imagefile in images, f"{PREF}- Incomplete set of images detected"

                # Hierarchical structure: [split] -> [class id] -> [images with class id]
                dirname = os.path.join(split, label)
                if not os.path.exists(dirname):
                    print(f"{PREF}- Creating class directory {dirname}")
                    os.mkdir(dirname)
                # Move file to new directory 
                shutil.copyfile(os.path.join(PATH, imagefile),
                                os.path.join(dirname, imagefile))

    print("[*] Successfully setup miniImageNet!")