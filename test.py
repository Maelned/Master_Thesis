import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# os.chdir("/home/ubuntu/Implementation_Mael")
dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ISIC2018V2\\"
training_dataset = dataset + "Training\\"
validation_dataset = dataset + "Validation\\"
test_dataset = dataset + "Test\\"
model = load_model("./Saves/Models/InceptionV3_v1.h5")
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

epsilon = 2/255
labels = {"akiec" : [[1.,0.,0.,0.,0.,0.,0.]],
            "bcc" : [[0.,1.,0.,0.,0.,0.,0.]],
            "bkl" : [[0.,0.,1.,0.,0.,0.,0.]],
            "df" :  [[0.,0.,0.,1.,0.,0.,0.]],
            "mel" : [[0.,0.,0.,0.,1.,0.,0.]],
            "nv"  : [[0.,0.,0.,0.,0.,1.,0.]],
            "vasc": [[0.,0.,0.,0.,0.,0.,1.]]
        }


test_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
)

test_ds = test_datagen.flow_from_directory(
    test_dataset,
    target_size=(299, 299),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=1,
    shuffle=False,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

def create_adversarial_pattern(input_image,input_label,model):
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

def FGSM_application():
    preds = []
    classes = [f for f in os.listdir(test_dataset)]

    for dir in classes:
        print("Dir = " + dir)
        current_dir = test_dataset + dir

        imgs = [i for i in os.listdir(current_dir)]

        for inames in imgs:

            img = tf.keras.preprocessing.image.load_img(os.path.join(current_dir, inames),
                                                                target_size=(224, 224))
            img = tf.keras.preprocessing.image.img_to_array(img)

            img = img.reshape([1,224, 224, 3])

            label = labels[dir]
            adv_noise = create_adversarial_pattern(img, label)
            noise = adv_noise * epsilon
            img_adv = img + noise

            #Test comparaison
            prediction = model.predict(img_adv)
            preds.append(list(prediction[0]))


    preds = list(preds)
    preds = np.argmax(preds, axis=1)
    cm_adv = confusion_matrix(test_ds.classes, preds)
    cm_adv = np.around(cm_adv, 2)
    print(cm_adv)
    print("FGSM Application Done")

FGSM_application()

Y_pred = model.predict_generator(test_ds, steps=test_ds.samples)
y_pred = np.argmax(Y_pred, axis=1)

cm_adv = confusion_matrix(test_ds.classes, y_pred)
cm_adv = np.around(cm_adv, 2)
print(cm_adv)