import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import UniversalPerturbation
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chestx')
parser.add_argument('--model', type=str, default='inceptionv3')
parser.add_argument('--norm', type=str, default='l2')
parser.add_argument('--eps', type=float, default=0.04)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

Validation_set = "E:\\DataSet\\ISIC2018\\ISIC_Validation\\"
model = load_model("./Saves/Models/InceptionV3_Model_2.h5")

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


def calculate_data(dataset, norm=False, ):
    if norm:
        mean_l2_train = 0
        mean_inf_train = 0
        for e in range(len(dataset)):
            i = next(dataset)
            image = i[0]
            mean_l2_train += np.linalg.norm(image[:, :, 0].flatten(), ord=2)
            mean_inf_train += np.abs(image[:, :, 0].flatten()).max()
        mean_l2_train /= len(dataset)
        mean_inf_train /= len(dataset)
        if norm:
            return mean_l2_train, mean_inf_train



def get_fooling_rate(preds, preds_adv):
    fooling_rate = np.sum(preds != preds_adv) / len(preds)
    return fooling_rate
print("calculation data")

# # Generate adversarial examples



classifier = KerasClassifier(model=model, use_logits=False)

adv_crafter = UniversalPerturbation(
    classifier=classifier,
    attacker='fgsm',
    attacker_params={'targeted': False, 'eps': 0.0024},
    max_iter=30,
    batch_size = 1,
    delta=0.000001)
print("adv_crafter created")
X_train = []
predict = []
Y_train = []
for e in range(len(val_ds)):
    i = next(val_ds)
    image = i[0]
    label = i[1]

    X_train.append(image[0])
    Y_train.append(label[0])

    pred = model.predict(image)

    prediction = list(pred)
    prediction = list(prediction)
    prediction_final = list(max(prediction))
    prediction_final = np.argmax(prediction_final)
    predict.append(prediction_final)

print(predict)
print("generate attack :")
#X_train = np.array(X_train)
print(np.shape(X_train))
_ = adv_crafter.generate(X_train)
prd = np.argmax(model.predict(val_ds,steps = val_ds.samples),axis = 1)
print(prd)
noise = adv_crafter.noise[0, :].astype(np.float32)

# # Evaluate the ART classifier on adversarial examples
print(np.shape(X_train))
prediction = np.argmax(classifier.predict(X_train), axis = 1)
print(prediction)
X_train_adv = X_train + noise

plt.imshow(X_train[0] *0.5+0.5)
plt.show()

plt.imshow(noise * 0.5 + 0.5)
plt.show()

plt.imshow(X_train_adv[0] *0.5+0.5)
plt.show()

prediction_adversarial = np.argmax(classifier.predict(X_train_adv), axis=1)

rf_train = get_fooling_rate(preds=prediction, preds_adv=prediction_adversarial)

print(rf_train)

print(prediction_adversarial)