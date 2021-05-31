import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import pickle
import webp
from os import listdir
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import UniversalPerturbation
from PIL import Image
import os,random,shutil
import numpy

labels = {"akiec" : [[1.,0.,0.,0.,0.,0.,0.]],
            "bcc" : [[0.,1.,0.,0.,0.,0.,0.]],
            "bkl" : [[0.,0.,1.,0.,0.,0.,0.]],
            "df" :  [[0.,0.,0.,1.,0.,0.,0.]],
            "mel" : [[0.,0.,0.,0.,1.,0.,0.]],
            "nv"  : [[0.,0.,0.,0.,0.,1.,0.]],
            "vasc": [[0.,0.,0.,0.,0.,0.,1.]]
        }



classe = ['akiec', 'bcc', 'bkl',
           'df', 'mel', 'nv', 'vasc']

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ISIC_dataset = "/mnt/data/Dataset/ISIC2018V2/Test/"
FGSM_dataset = "/mnt/data/Dataset/LLT_Datasets/UAP/"
#Model trained

model = load_model("./Saves/Models/Retrained_model_v3_5epoch_5times.h5")
model = load_model("./Saves/Models/InceptionV3_v3.h5")
loss_object = tf.keras.losses.CategoricalCrossentropy()

epsilon = 2/255.


val_datagen_test = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
)

val_ds_test= val_datagen_test.flow_from_directory(
    ISIC_dataset,
    target_size=(299, 299),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=1,
    shuffle=False,
    seed=False,
    interpolation="bilinear",
    follow_links=False)


def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        # explicitly indicate that our image should be tacked for
        # gradient updates
        tape.watch(input_image)
        # use our model to make predictions on the input image and
        # then compute the loss
        pred = model(input_image)
        loss = loss_object(input_label, pred)
        # calculate the gradients of loss with respect to the image, then
        # compute the sign of the gradient
        gradient = tape.gradient(loss, input_image)
        signedGrad = tf.sign(gradient)

        return signedGrad


def FGSM_application():
    classifier = KerasClassifier(model=model, use_logits=False)

    adv_crafter = UniversalPerturbation(
        classifier=classifier,
        attacker='fgsm',
        attacker_params={'targeted': False, 'eps': 0.0024},
        max_iter=10,
        batch_size=1,
        delta=0.000001)
    X_train = []
    Y_train = []
    for e in range(len(val_ds_test)):
        i = next(val_ds_test)
        image = i[0]
        label = i[1]

        X_train.append(image[0])
        Y_train.append(label[0])

    print("generate attack :")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    _ = adv_crafter.generate(X_train, Y_train)
    noise = adv_crafter.noise[0, :].astype(np.float32)
    X_train_adv = X_train + noise
    os.chdir(FGSM_dataset)
    preds = []
    for i in range(len(X_train_adv)):
        image = X_train_adv[i]
        label = Y_train[i]
        dir = classe[np.argmax(label)]
        os.makedirs(dir, exist_ok=True)
        os.chdir(FGSM_dataset + dir + "/")

        prediction = model.predict(image)
        preds.append(prediction[0])
        img_adv = np.fliplr(image)
        img_adv = tf.keras.preprocessing.image.array_to_img(img_adv[0])
        name = "{}".format(e)
        webp.save_image(img_adv, "/mnt/data/Dataset/LLT_Datasets/FGSM/" + dir + "/" + name + ".webp", quality=100)
        os.chdir(FGSM_dataset)
    preds = list(preds)
    preds = np.argmax(preds, axis=1)
    cm_adv = confusion_matrix(val_ds_test.classes, preds)
    cm_adv = np.around(cm_adv, 2)
    print(cm_adv)
    return cm_adv

cm_adv = FGSM_application()

Modified_dataset = "/mnt/data/Dataset/LLT_Datasets/FGSM/"

classes = listdir(Modified_dataset)
classes = sorted(classes)
preds = []
for current_class in classes:
    current_dir = Modified_dataset + current_class + "/"
    imgs = [i for i in os.listdir(current_dir)]
    imgs = sorted(imgs)
    for current_img in imgs:
        img = webp.load_image(os.path.join(current_dir, current_img), 'RGB')

        img = np.asarray(img)
        img = img / 255.
        img = img.reshape([1, 299, 299, 3])
        prediction = model.predict(img)

        preds.append(prediction[0])

preds = np.argmax(preds, axis = 1)
cm = confusion_matrix(val_ds_test.classes, preds)
cm = np.around(cm, 2)
print(cm)
name_cm = "/home/ubuntu/Implementation_Mael/pythonProject1/Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_FGSM_Compressed_Flipped.pkl"
with open(name_cm, 'wb') as f:
    pickle.dump(cm, f)