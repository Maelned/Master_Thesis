import operator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict


dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ISIC2018V2\\"
training_dataset = dataset + "Training\\"
validation_dataset = dataset + "Validation\\"
test_dataset = dataset + "Test\\"
data_dir = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ISIC2018\\"

label = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
Healthy_or_not = [0, 0, 1, 1, 0, 0, 1]
Healthy_counter = 0
Cancerous_counter = 0
datagen = ImageDataGenerator(
    rescale=1. / 255.,
)

dataset_training = datagen.flow_from_directory(
    training_dataset,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)
dataset_validation = datagen.flow_from_directory(
    validation_dataset,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)
dataset_test = datagen.flow_from_directory(
    test_dataset,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

Labels_count_training = Counter(dataset_training.classes)
Labels_count_validation = Counter(dataset_validation.classes)
Labels_count_test = Counter(dataset_test.classes)

Labels_count_total = [Labels_count_training,Labels_count_validation,Labels_count_test]

for Labels_count in Labels_count_total:
    Label_sorted = sorted(Labels_count.items(), key=operator.itemgetter(0),reverse=True)
    # print(Label_sorted)
    x,y = zip(*Label_sorted)
    x = list(x)
    y = list(y)
    a = 0
    for i in x:
        x[a] = label[i]
        if Healthy_or_not[i]:
            Healthy_counter += y[a]
        else:
            Cancerous_counter += y[a]
        a += 1

print(Healthy_counter)
print(Cancerous_counter)

print(Healthy_counter + Cancerous_counter)
Labels = ["Healthy","Cancerous"]
data = [Healthy_counter,Cancerous_counter]
color = ["tab:green","tab:red"]
explode = [ 0.05,0.05]
# Creating plot
fig = plt.figure(figsize=(6, 6))
plt.pie(data, labels=Labels,explode = explode,colors = color, autopct="%.1f%%",textprops={'fontsize': 14})

# show plot
plt.show()