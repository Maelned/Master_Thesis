import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import operator

Validation_set = "E:\\DataSet\\ISIC2018\\ISIC_Validation\\"
new_model = load_model("./Saves/Models/InceptionV3_Model_2.h5")

loss_object = tf.keras.losses.CategoricalCrossentropy()



def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        # explicitly indicate that our image should be tacked for
        # gradient updates
        tape.watch(input_image)
        # use our model to make predictions on the input image and
        # then compute the loss
        pred = new_model(input_image)
        loss = loss_object(input_label, pred)
        # calculate the gradients of loss with respect to the image, then
        # compute the sign of the gradient
        gradient = tape.gradient(loss, input_image)
        signedGrad = tf.sign(gradient)

        return signedGrad

val_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
)

val_ds = val_datagen.flow_from_directory(
    Validation_set,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=1,
    shuffle=False,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

eps=2 / 255.0
print("TAILLE DE DATASET :",len(val_ds))
Y_pred = new_model.predict_generator(val_ds, steps = val_ds.samples)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(val_ds.classes,y_pred)
cm = np.around(cm,2)
eps=2 / 255.0

classes = ['actinic keratoses', 'basal cell carcinoma', 'benign keratosis-like lesions',
           'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesions']
batch=next(val_ds)  # returns the next batch of images and labels

pred = new_model.predict(batch[0])
img_adv = create_adversarial_pattern(batch[0],batch[1])
pred_adv = new_model.predict(img_adv)
print(pred_adv)
preds = []
test = 0
for e in range(193):
    eps = [0,0.01,0.1,0.15, 2/255.0]
    i = next(val_ds)
    image = i[0]
    label = i[1]
    adv_noise = create_adversarial_pattern(image, label)
    # construct the image adversary
    for ep in eps:
        img_adv = (image + (adv_noise * ep)).numpy()
        # return the image adversary to the calling function
        true_label = classes[np.argmax(label, axis=1)[0]]

        prediction = new_model.predict(img_adv)
        prediction = list(prediction)
        prediction = list(max(prediction))
        confidence = max(prediction)

        label2 = classes[prediction.index(confidence)]
        noise = adv_noise * ep
        im_adv = image + noise
        im_adv = tf.clip_by_value(im_adv, -1, 1)
        plt.title("Noise added with eps :%1.3f" %ep)
        plt.imshow(noise[0])
        plt.show()
        plt.title("True label : " + true_label + "\nPredicted label and confidence :  %1.3f " %confidence +" "+ label2)
        plt.imshow(im_adv[0] * 0.5 + 0.5)
        plt.show()

    prediction = new_model.predict(img_adv)
    preds.append(list(prediction[0]))

print("sortie for")

preds= list(preds)

preds = np.argmax(preds, axis=1)
cm_adv = confusion_matrix(val_ds.classes, preds)
cm_adv = np.around(cm_adv,2)
print(cm_adv)


with open("./Saves/ConfusionMatrixes/ConfusionMatrix_BeforeFGSM.pkl", 'wb') as f:
    pickle.dump(cm, f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_AfterFGSM.pkl", 'wb') as f:
    pickle.dump(cm_adv, f)





