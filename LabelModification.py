import os, random, shutil
from os import listdir


Dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV3\\"
num_ref = len(listdir("E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV3\\05 nv-bkl\\Validation\\nv"))

# Dataset = "/mnt/data/Dataset/ModifiedLabelsV2/"
# num_ref = len(listdir("/mnt/data/Dataset/ModifiedLabelsV2/05 nv-bkl/Validation/nv"))

# current_dir = Dataset + "\\Validation\\nv"
# dest_dir = Dataset + "\\Validation\\bkl"
# filenames = random.sample(os.listdir(current_dir), int(am))
# for fname in filenames:
#     srcpath = current_dir + "\\" + fname.strip()
#     dest_path = dest_dir + "\\" + fname.strip()
#     os.replace(srcpath,dest_path)

amount = [0.05,0.10,0.15,0.20,0.30,0.45,0.60]
experiment = listdir("E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV3\\")
experiment = sorted(experiment)
am = amount * num_ref

for i in range(len(amount)):
    am = amount[i] * num_ref
    if i:
        am = (amount[i] - amount[i-1]) * num_ref
    dir = experiment[i]
    other_dir = experiment[i:]
    current_dir = Dataset + dir + "\\Validation\\nv"
    filenames = random.sample(os.listdir(current_dir), int(am))
    for current_exp in other_dir:
        current_dir = Dataset + current_exp + "\\Validation\\nv"
        dest_dir = Dataset + current_exp + "\\Validation\\bkl"
        for fname in filenames:
            srcpath = current_dir + "\\" + fname.strip()
            dest_path = dest_dir + "\\" + fname.strip()
            os.replace(srcpath,dest_path)
