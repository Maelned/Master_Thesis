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
new_model = load_model("Saves/Models/InceptionV3.h5")

loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)



def create_adversarial_pattern(input_image,input_label):
    with tf.GradientTape() as tape:

        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        # explicitly indicate that our image should be tacked for
        plt.imshow(input_image[0])
        plt.show()
        # gradient updates
        tape.watch(input_image)
        # use our model to make predictions on the input image and
        # then compute the loss
        pred = new_model(input_image)
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

# classifier = KerasClassifier(model=new_model, use_logits=False)

# print("adv_crafter created")
# X_train = []
# Y_train = []
# for e in range(len(test_ds)):
#     i = next(test_ds)
#     image = i[0]
#     label = i[1]
#
#     X_train.append(image[0])
#     Y_train.append(label[0])
epsilon = [0,0.007,2/255.0,0.01,3/255,0.05,0.1]
# X_train = np.array(X_train)

for eps in epsilon:
    print("Epsilon : ", eps)
    # attack = FastGradientMethod(estimator=classifier, eps=eps)
    # x_test_adv = attack.generate(x=X_train)
    # predictions = classifier.predict(x_test_adv)
    # accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(Y_train, axis=1)) / len(Y_train)
    # print("Accuracy on adversarial test examples: {}%".format(accuracy * 100)
    preds = []
    for e in range(len(test_ds)):
        i = next(test_ds)
        image = i[0]
        label = i[1]
        plt.imshow(image[0])
        plt.show()
        adv_noise = create_adversarial_pattern(image,label)

        # construct the image adversary
        img_adv = (image + (adv_noise * eps))
        plt.imshow(img_adv[0])
        plt.show()
        # if not e:
        #     plt.title("Adversarial image")
        #     plt.imshow(img_adv[0])
        #     plt.show()
        #     plt.title("Adversarial noise without  numpy")
        #     plt.imshow((adv_noise * eps)[0])
        #     plt.show()
        # return the image adversary to the calling function
        img_adv= tf.clip_by_value(img_adv, -1, 1)
        prediction = new_model.predict(img_adv)
        preds.append(list(prediction[0]))

    preds = list(preds)
    preds = np.argmax(preds, axis=1)
    cm_adv = confusion_matrix(test_ds.classes, preds)
    cm_adv = np.around(cm_adv, 2)
    print(cm_adv)

    with open("./Saves/ConfusionMatrixes/ConfusionMatrix_AfterFGSM_InceptionV3_Version2_{}.pkl".format(eps), 'wb') as f:
        pickle.dump(cm_adv, f)








