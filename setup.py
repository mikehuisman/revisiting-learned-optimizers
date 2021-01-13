import os

PREF = " "*5
# Try to import gdown, else download it
try:
    print("[*] Verifying whether `gdown' has been installed...")
    import gdown
    print(f"{PREF}- Working installation found.")
except:
    print("[*] Failed to find `gdown' package. Installing now...")
    os.system("pip install gdown")
    print(f"{PREF}- Installation succesful!")

from data.cub.setup_cub import setup_cub
from data.min.setup_min import setup_min


# String to separate std output
cut = "-"*40

# Setup miniImageNet
print(cut)
print(f"[@] Working on miniImageNet:")
os.chdir("./data/min/")
setup_min()
print(cut)

# Setup CUB 
print("\n\n")
print(cut)
print(f"[@] Working on CUB:")
os.chdir("../cub/")
setup_cub()
print(cut)

# Print completion
print("\n\n")
print(cut)
print("[@] miniImageNet and CUB ready for use.")
os.chdir("../../")
print(cut)