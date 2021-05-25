import numpy as np
import pickle
import os
import random
import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import UniversalPerturbation
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
from sklearn.metrics import confusion_matrix
tf.compat.v1.disable_eager_execution()
os.chdir("/home/ubuntu/Implementation_Mael/pythonProject1/")

dataset = "/mnt/data/Dataset/ISIC2018V2/"
training_dataset = dataset + "Training/"
validation_dataset = dataset + "Validation/"
Test_dataset = dataset + "Test/"

base_model = load_model("Saves/Models/InceptionV3_v3.h5")
number_times = 5
nb_epochs = 10
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
    classifier = KerasClassifier(model=model, use_logits=False)

    adv_crafter = UniversalPerturbation(
        classifier=classifier,
        attacker='fgsm',
        attacker_params={'targeted': False, 'eps': 0.0024},
        max_iter=10,
        batch_size=1,
        delta=0.000001)

    print("Adversarial Training in progress")
    X_train_sample, Y_train_sample = [], []
    X_val_sample, Y_val_sample = [], []
    X_train, Y_train = [], []
    X_val, Y_val = [], []

    am_train = int(len(train) * amount)
    am_val = int(len(val) * amount)
    random_samples_train = random.sample(train,am_train)
    random_samples_val = random.sample(val, am_val)

    for current_sample in random_samples_train:
        label = current_sample[1]
        image = current_sample[0]
        X_train_sample.append(image[0])
        Y_train_sample.append(label[0])

    _ = adv_crafter.generate(np.array(X_train_sample), np.array(Y_train_sample))
    noise = adv_crafter.noise[0, :].astype(np.float32)
    X_train_adv_sample = X_train_sample + noise

    for i in train:
        image = i[0]
        label = i[1]
        X_train.append(image[0])
        Y_train.append(label[0])

    X_train_adv = np.concatenate((X_train,X_train_adv_sample))
    Y_train_adv = np.concatenate((Y_train,Y_train_sample))


    for current_sample in random_samples_val:
        label = current_sample[1]
        image = current_sample[0]
        X_val_sample.append(image[0])
        Y_val_sample.append(label[0])

    _ = adv_crafter.generate(np.array(X_val_sample), np.array(Y_val_sample))
    noise = adv_crafter.noise[0, :].astype(np.float32)
    X_val_adv_sample = X_val_sample + noise

    for i in val:
        image = i[0]
        label = i[1]
        X_val.append(image[0])
        Y_val.append(label[0])

    X_val_adv = np.concatenate((X_val,X_val_adv_sample))
    Y_val_adv = np.concatenate((Y_val,Y_val_sample))

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
        model = load_model("./Saves/Models/Retrained_model_v3_UAP_10epoch_{}times.h5".format(i))

    X_train_adv, Y_train_adv, X_val_adv, Y_val_adv = adversarialTraining(train, val, 0.5)

    X_train_adv = np.array([x for x in X_train_adv])
    Y_train_adv = np.array([x for x in Y_train_adv])
    X_val_adv = np.array([x for x in X_val_adv])
    Y_val_adv = np.array([x for x in Y_val_adv])

    class_weights = class_weight.compute_class_weight("balanced",
                                                      np.unique(np.argmax(Y_train_adv, axis=1)),
                                                      np.argmax(Y_train_adv, axis=1))
    class_weights = {i: class_weights[i] for i in range(7)}

    model.compile(optimizer=SGD(lr=7e-5,momentum=0.9), loss="categorical_crossentropy", metrics=[categorical_accuracy])
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

    name_model = "./Saves/Models/Retrained_model_v3_UAP_10epoch_{}times.h5".format(i+1)

    model.save(name_model)