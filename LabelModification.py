import os, random, shutil


pwd = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabels\\75% nv - bkl"

current_dir = pwd + "\\nv"
dest_dir = pwd + "\\bkl"

filenames = random.sample(os.listdir(current_dir), 5025)
for fname in filenames:
    srcpath = current_dir + "\\" + fname.strip()
    dest_path = dest_dir + "\\" + fname.strip()
    #shutil.copyfile(srcpath, dest_dir)
    os.replace(srcpath,dest_path)
