from os import listdir
import os,random

Dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\"
ISIC = Dataset + "ISIC2018\\"

Training_dir = Dataset + "ISIC2018V2\\Training\\"
Validation_dir = Dataset + "ISIC2018V2\\Validation\\"
Test_dir = Dataset + "ISIC2018V2\\Test\\"

training = 0.7
validation = 0.175
amount = [0.7,0.6,1]

classes = [f for f in listdir(ISIC)]

for dir in classes:
    current_dir = ISIC + dir
    for nb in amount:

        nb_img = len(os.listdir(current_dir))
        filenames = random.sample(os.listdir(current_dir),  int((nb * nb_img)))
        if nb == 0.7:
            for fname in filenames:
                srcpath = current_dir + "\\" + fname.strip()
                dest_path = Training_dir + dir + "\\" + fname.strip()
                os.replace(srcpath, dest_path)
        elif nb == 0.6:
            for fname in filenames:
                srcpath = current_dir + "\\" + fname.strip()
                dest_path = Validation_dir + dir + "\\" + fname.strip()
                os.replace(srcpath, dest_path)
        else:
            for fname in filenames:
                srcpath = current_dir + "\\" + fname.strip()
                dest_path = Test_dir + dir + "\\" + fname.strip()
                os.replace(srcpath, dest_path)

print("Dataset split")