import os, random, shutil
from os import listdir


Dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV2\\"
num_ref = len(listdir("E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ISIC2018V2\\Training\\nv"))

experiment = [f for f in listdir(Dataset)]
amount = [0.05,0.10,0.15,0.20,0.30,0.45,0.60,0.75]

for i in range(len(amount)):
    am = amount[i] * num_ref
    dir = experiment[i]
    current_dir = Dataset + dir + "\\Training\\nv"
    dest_dir = Dataset + dir + "\\Training\\bkl"
    filenames = random.sample(os.listdir(current_dir), int(am))
    for fname in filenames:
        srcpath = current_dir + "\\" + fname.strip()
        dest_path = dest_dir + "\\" + fname.strip()
        os.replace(srcpath,dest_path)
