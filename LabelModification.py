import os, random, shutil
from os import listdir


Dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV2\\60 nv-bkl"
num_ref = len(listdir("E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV2\\60 nv-bkl\\Training\\nv"))


amount = 0.60
am = amount * num_ref
current_dir = Dataset + "\\Training\\nv"
dest_dir = Dataset + "\\Training\\bkl"
filenames = random.sample(os.listdir(current_dir), int(am))
for fname in filenames:
    srcpath = current_dir + "\\" + fname.strip()
    dest_path = dest_dir + "\\" + fname.strip()
    os.replace(srcpath,dest_path)

# for i in range(len(amount)):
#     am = amount[i] * num_ref
#     dir = experiment[i]
#     current_dir = Dataset + dir + "\\Training\\nv"
#     dest_dir = Dataset + dir + "\\Training\\bkl"
#     filenames = random.sample(os.listdir(current_dir), int(am))
#     for fname in filenames:
#         srcpath = current_dir + "\\" + fname.strip()
#         dest_path = dest_dir + "\\" + fname.strip()
#         os.replace(srcpath,dest_path)
