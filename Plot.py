import operator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict

data_dir = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\HAM10K\\"
data_dir5 = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV2\\05 nv-bkl\\"
data_dir10 = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV2\\10 nv-bkl\\"
data_dir15 = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV2\\15 nv-bkl\\"
data_dir20 = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabelsV2\\20 nv-bkl\\"
label = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

datagen = ImageDataGenerator(
    rescale=1. / 255.,
)

dataset5 = datagen.flow_from_directory(
    data_dir5,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

dataset10 = datagen.flow_from_directory(
    data_dir10,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

dataset15 = datagen.flow_from_directory(
    data_dir15,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

dataset20 = datagen.flow_from_directory(
    data_dir20,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

dataset = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)



Labels_count = Counter(dataset5.classes)
Label_sorted = sorted(Labels_count.items(), key=operator.itemgetter(1),reverse=True)
x,y = zip(*Label_sorted)

print(x,y)
x = list(x)
Modified = [0,335,0,0,0,0,0]
y = list(y)
y[1] = y[1] - 335
a = 0
for i in x:
    x[a] = label[i]
    a += 1

print(x)
index = range(len(x))
index2 = range(len(y))
plt.ylim([0,6800])
my_colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
# s = pd.Series(y, index=x)
#
# s.plot(kind='bar',rot=0)
plt.bar(x,y,color=my_colors,width=0.5)
plt.bar(x,Modified,color = "tab:blue", bottom=y,width=0.5)
plt.title('Classes distribution with 5% of NV in BKL')
plt.show()
