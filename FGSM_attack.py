import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
import pickle
import matplotlib.pyplot as plt
# tf.compat.v1.disable_eager_execution()

Test_set = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ISIC2018V2\\Test\\"

model_60fgsm = load_model("Saves/Models/InceptionV3_AdversarialTraining_60fgsm.h5")

loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
name_model = ["60fgsm"]
models = [model_60fgsm]

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

test_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
)

test_ds = test_datagen.flow_from_directory(
    Test_set,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=1,
    shuffle=False,
    seed=False,
    interpolation="bilinear",
    follow_links=False)



classes = ['actinic keratoses', 'basal cell carcinoma', 'benign keratosis-like lesions',
           'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesions']


eps = 2 / 255.0
print("Epsilon : ", eps)

# attack = FastGradientMethod(estimator=classifier, eps=eps)
# x_test_adv = attack.generate(x=X_train)
# predictions = classifier.predict(x_test_adv)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(Y_train, axis=1)) / len(Y_train)
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100)

for model in models:
    preds = []
    for e in range(len(test_ds)):
        i = next(test_ds)
        image = i[0]
        label = i[1]
        adv_noise = create_adversarial_pattern(image,label,model)
        # construct the image adversary
        img_adv = (image + (adv_noise * eps))
        img_adv= tf.clip_by_value(img_adv, -1, 1)
        prediction = model.predict(img_adv)
        preds.append(prediction[0])

    preds = list(preds)
    preds = np.argmax(preds, axis=1)
    cm_adv = confusion_matrix(test_ds.classes, preds)
    cm_adv = np.around(cm_adv, 2)
    print(cm_adv)
    index_model = models.index(model)
    model_name = name_model[index_model]

    name_cm = "./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_FGSM_After_Adversarial_Training_{}.pkl".format(model_name)
    with open(name_cm, 'wb') as f:
        pickle.dump(cm_adv, f)








