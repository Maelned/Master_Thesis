import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import UniversalPerturbation
from sklearn.metrics import confusion_matrix
import pickle
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chestx')
parser.add_argument('--model', type=str, default='inceptionv3')
parser.add_argument('--norm', type=str, default='l2')
parser.add_argument('--eps', type=float, default=0.04)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

Test_set = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ISIC2018V2\\Test\\"
model = load_model("./Saves/Models/Retrained_model_v6_5epoch_5times.h5")

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
    target_size=(299, 299),
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
    max_iter=10,
    batch_size = 1,
    delta=0.000001)
print("adv_crafter created")
X_train = []
predict = []
Y_train = []
for e in range(len(test_ds)):
    i = next(test_ds)
    image = i[0]
    label = i[1]

    X_train.append(image[0])
    Y_train.append(label[0])


print("generate attack :")
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(np.shape(X_train))
_ = adv_crafter.generate(X_train,Y_train)
prd = np.argmax(model.predict(test_ds,steps = test_ds.samples),axis = 1)
print(prd)
noise = adv_crafter.noise[0, :].astype(np.float32)

# # Evaluate the ART classifier on adversarial examples
print(np.shape(X_train))
prediction = np.argmax(classifier.predict(X_train), axis = 1)
print(prediction)
X_train_adv = X_train + noise

prediction_adversarial = np.argmax(classifier.predict(X_train_adv), axis=1)

cm_adv = confusion_matrix(test_ds.classes, prediction_adversarial)
cm_adv = np.around(cm_adv, 2)
print(cm_adv)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_NonTargetedUAP_RetrainedModel_v6.pkl", 'wb') as f:
    pickle.dump(cm_adv, f)

rf_train = get_fooling_rate(preds=prediction, preds_adv=prediction_adversarial)

print(rf_train)

print(prediction_adversarial)