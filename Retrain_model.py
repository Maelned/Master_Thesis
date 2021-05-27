import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import itertools
import random
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
from sklearn.metrics import confusion_matrix

os.chdir("/home/ubuntu/Implementation_Mael/pythonProject1/")

dataset = "/mnt/data/Dataset/ISIC2018V2/"
training_dataset = dataset + "Training/"
validation_dataset = dataset + "Validation/"
Test_dataset = dataset + "Test/"

# base_model = load_model("/home/ubuntu/Implementation_Kentin/Perf/ResNetV2.h5")
base_model = load_model("/home/ubuntu/Implementation_Mael/pythonProject1/Saves/Models/InceptionV3_v3.h5")
number_times = 7
nb_epochs = 5
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

# new line
def create_adversarial_pattern(input_image,input_label, model):
    with tf.GradientTape() as tape:

        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        # explicitly indicate that our image should be tacked for
        # gradient updates
        tape.watch(input_image)
        # use our model to make predictions on the input image and
        # then compute the loss
        pred = model(input_image)
        loss = loss_object(input_label,pred)
        # calculate the gradients of loss with respect to the image, then
        # compute the sign of the gradient
        gradient = tape.gradient(loss, input_image)
        signedGrad = tf.sign(gradient)

        return signedGrad

def get_dataset(train_ds, val_ds):
    train, val = [], []

    for e in range(len(train_ds)):
        i = next(train_ds)
        image = i[0]
        label = i[1]
        tmp = (image,label)
        train.append(tmp)

    for e in range(len(val_ds)):
        i = next(val_ds)
        image = i[0]
        label = i[1]
        tmp = (image, label)
        val.append(tmp)
    return train, val


def adversarialTraining(train, val, amount):
    print("Adversarial Training in progress")
    X_train_adv, Y_train_adv, X_val_adv, Y_val_adv = [], [], [], []

    am_train = int(len(train) * amount)
    am_val = int(len(val) * amount)
    random_samples_train = random.sample(train,am_train)
    random_samples_val = random.sample(val, am_val)

    for i in train:
        image = i[0]
        label = i[1]
        X_train_adv.append(image[0])
        Y_train_adv.append(label[0])

    for i in val:
        image = i[0]
        label = i[1]
        X_val_adv.append(image[0])
        Y_val_adv.append(label[0])
    preds_normal = []
    label_normal = []
    preds_adv = []
    for current_sample in random_samples_train:
        label = current_sample[1]
        image = current_sample[0]
        prediction = base_model.predict(image)
        preds_normal.append(prediction[0])
        label_normal.append(label[0])
        adv_noise = create_adversarial_pattern(image, label,base_model)
        # construct the image adversary
        img_adv = (image + (adv_noise * 2/255.))
        prediction = base_model.predict(img_adv)
        preds_adv.append(prediction[0])
        X_train_adv.append(img_adv[0])
        Y_train_adv.append(label[0])

    for current_sample in random_samples_val:
        label = current_sample[1]
        image = current_sample[0]
        adv_noise = create_adversarial_pattern(image, label,base_model)
        # construct the image adversary
        img_adv = (image + (adv_noise * 2/255.))
        X_val_adv.append(img_adv[0])
        Y_val_adv.append(label[0])

    return X_train_adv, Y_train_adv, X_val_adv, Y_val_adv

batch_size = 1
train_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False,
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
)

train_ds = train_datagen.flow_from_directory(
    training_dataset,
    target_size=(299, 299),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

val_ds = val_datagen.flow_from_directory(
    validation_dataset,
    target_size=(299, 299),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)


test_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
)

test_ds = test_datagen.flow_from_directory(
    Test_dataset,
    target_size=(299, 299),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=1,
    shuffle=False,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

batch_size = 32
learning_rate_reduction = ReduceLROnPlateau(monitor='val_categorical_accuracy',
                                            patience=5,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.00001)



train, val = get_dataset(train_ds, val_ds)
eps = 2/255.0

for i in range(number_times):
    print("Retrain model {} times".format(i+1))
    preds = []
    if i == 0:
        print("base_model")
        model = base_model
    else:
        print("changing model")
        model = load_model("./Saves/Models/Retrained_model_InceptionV3_5epoch_{}times.h5".format(i))
    X_train_adv, Y_train_adv, X_val_adv, Y_val_adv = adversarialTraining(train, val, 0.5)
    X_train_adv = np.array([x for x in X_train_adv])
    Y_train_adv = np.array([x for x in Y_train_adv])
    X_val_adv = np.array([x for x in X_val_adv])
    Y_val_adv = np.array([x for x in Y_val_adv])

    class_weights = class_weight.compute_class_weight("balanced",
                                                      np.unique(np.argmax(Y_train_adv, axis=1)),
                                                      np.argmax(Y_train_adv, axis=1))
    class_weights = {i: class_weights[i] for i in range(7)}

    model.compile(optimizer=SGD(lr=9e-5,momentum=0.9), loss="categorical_crossentropy", metrics=[categorical_accuracy])
    history = model.fit(
        x=X_train_adv,
        y=Y_train_adv,
        epochs=nb_epochs,
        batch_size = 32,
        validation_data= (X_val_adv, Y_val_adv),
        class_weight= class_weights,
        shuffle = True,
        verbose=2,
        workers=8,
        callbacks=[learning_rate_reduction]
    )

    name_model = "./Saves/Models/Retrained_model_InceptionV3_5epoch_{}times.h5".format(i+1)
    model.save(name_model)

    print("before test loop")
    for e in range(len(test_ds)):
        a = next(test_ds)
        image = a[0]
        label = a[1]
        adv_noise = create_adversarial_pattern(image,label,model)
        # construct the image adversary
        img_adv = (image + (adv_noise * eps))
        # img_adv = tf.clip_by_value(img_adv, -1, 1)

        prediction = model.predict(img_adv)
        preds.append(prediction[0])

    preds = list(preds)
    preds = np.argmax(preds, axis=1)
    cm_adv = confusion_matrix(test_ds.classes, preds)
    cm_adv = np.around(cm_adv, 2)
    print(cm_adv)

    name_cm = "./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_FGSM_Retrained_Model_5epochs_{}times.pkl".format(i+1)
    with open(name_cm, 'wb') as f:
        pickle.dump(cm_adv, f)