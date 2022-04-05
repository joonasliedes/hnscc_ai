from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
from sklearn.model_selection import train_test_split
import imageFunctions
import lossFunctions
import random
import augmentationFunctions
from dataToExcel import loadToExcel
import evaluationFunctions
import matplotlib.pyplot as plt
from models import create_unet
from sklearn.utils import shuffle
import segmentation_models as sm
import cv2

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 2
SEED = 1
random.seed(SEED)

# Get patients with cancer
pos_patients_pet = imageFunctions.getPatients("pet", "pos")
pos_patients_mri = imageFunctions.getPatients("mri", "pos")
pos_patients_masks = imageFunctions.getPatients("mask", "pos")

# Fuse positives
fused_patients_pos = imageFunctions.fuseImages(pos_patients_mri, pos_patients_pet)

# remove images without cancer from patient stacks
inds = []
fused_patients_cancer = []
masks_cancer = []
for i in range(len(fused_patients_pos)):
    for j in range(fused_patients_pos[i].shape[2]):
        if np.max(pos_patients_masks[i][:, :, j]) == 0:
            inds.append(j)
    fused_arr = np.delete(fused_patients_pos[i], obj=inds, axis=2)
    masks_arr = np.delete(pos_patients_masks[i], obj=inds, axis=2)
    fused_patients_cancer.append(fused_arr)
    masks_cancer.append(masks_arr)
    inds.clear()
# Get patients without cancer
neg_patients_pet = imageFunctions.getPatients("pet", "neg")
neg_patients_mri = imageFunctions.getPatients("mri", "neg")


# Fuse negatives
fused_patients_neg = imageFunctions.fuseImages(neg_patients_mri, neg_patients_pet)

# Merge fused
all_fused = fused_patients_cancer + fused_patients_neg


# negative masks
neg_patient_masks = []
for i in range(len(fused_patients_neg)):
    neg_mask = np.zeros(shape=(512, 512, fused_patients_neg[i].shape[2]))
    neg_patient_masks.append(neg_mask)


all_masks = masks_cancer + neg_patient_masks

# Divide patientwise
stratify_val = [0 if mask.any() else 1 for mask in all_masks]

train_ids, test_ids, train_labels, test_labels = train_test_split(
    all_fused, all_masks, test_size=0.20, random_state=SEED, stratify=stratify_val
)

# Separate positives and negatives and random sample negatives to match the amount of positives

neg_train_ids = []
neg_train_labels = []
pos_train_ids = []
pos_train_labels = []
for i in range(len(train_ids)):
    if np.max(train_labels[i]) == 0:
        neg_train_ids.append(train_ids[i])
        neg_train_labels.append(train_labels[i])
    else:
        pos_train_ids.append(train_ids[i])
        pos_train_labels.append(train_labels[i])
neg_train_slices = imageFunctions.getSlices(neg_train_ids)
pos_train_slices = imageFunctions.getSlices(pos_train_ids)
pos_train_mask_slices = imageFunctions.getMaskSlices(pos_train_labels)
neg_train_mask_slices = []
rand_neg_train_slices = random.sample(neg_train_slices, len(pos_train_slices))
for i in range(len(rand_neg_train_slices)):
    neg_mask_slice = np.zeros(shape=(512, 512))
    neg_train_mask_slices.append(neg_mask_slice)


train_slices = pos_train_slices + rand_neg_train_slices
train_mask_slices = pos_train_mask_slices + neg_train_mask_slices


neg_test_ids = []
neg_test_labels = []
pos_test_ids = []
pos_test_labels = []
for i in range(len(test_ids)):
    if np.max(test_labels[i]) == 0:
        neg_test_ids.append(test_ids[i])
        neg_test_labels.append(test_labels[i])
    else:
        pos_test_ids.append(test_ids[i])
        pos_test_labels.append(test_labels[i])
neg_test_slices = imageFunctions.getSlices(neg_test_ids)
pos_test_slices = imageFunctions.getSlices(pos_test_ids)
pos_test_mask_slices = imageFunctions.getMaskSlices(pos_test_labels)
neg_test_mask_slices = []

rand_neg_test_slices = random.sample(neg_test_slices, len(pos_test_slices))
for i in range(len(rand_neg_test_slices)):
    neg_mask_slice = np.zeros(shape=(512, 512))
    neg_test_mask_slices.append(neg_mask_slice)


test_slices = pos_test_slices + rand_neg_test_slices
test_mask_slices = pos_test_mask_slices + neg_test_mask_slices


# shuffle slices
train_slices, train_mask_slices = shuffle(
    train_slices, train_mask_slices, random_state=SEED
)
test_slices, test_mask_slices = shuffle(
    test_slices, test_mask_slices, random_state=SEED
)
# Resize and normalize
print("Resizing images")

for i in range(len(train_slices)):
    img = cv2.resize(
        train_slices[i], (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC
    )
    train_slices[i] = img

print("Resizing masks")

for i in range(len(train_mask_slices)):
    mask = cv2.resize(
        train_mask_slices[i], (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC
    )
    train_mask_slices[i] = mask

print("Normalising PET and MRI images")
mri_min, mri_max, pet_min, pet_max = imageFunctions.getMinMax(
    train_slices
)  # Get min and max values


imageFunctions.normalize(train_slices, mri_min, mri_max, pet_min, pet_max)

X_train = np.array(train_slices, dtype=np.float32)
Y_train = np.array(train_mask_slices, dtype=bool)

# X_train = np.expand_dims(X_train, axis=3)
Y_train = np.expand_dims(Y_train, axis=3)


print("DONE")

print("Resizing test images")

for i in range(len(test_slices)):
    img = cv2.resize(
        test_slices[i], (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC
    )
    test_slices[i] = img
print("resising test masks")
for i in range(len(test_mask_slices)):
    mask = cv2.resize(
        test_mask_slices[i], (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC
    )
    test_mask_slices[i] = mask
print("Normalising test images")

imageFunctions.normalize(test_slices, mri_min, mri_max, pet_min, pet_max)

X_test = np.array(test_slices, dtype=np.float32)
Y_test = np.array(test_mask_slices, dtype=bool)
# X_test = np.expand_dims(X_test, axis=3)
Y_test = np.expand_dims(Y_test, axis=3)
print("DONE")

print("Augmenting images")
X_train_aug, Y_train_aug = augmentationFunctions.augment(
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 5, X_train, Y_train, SEED
)
print("DONE")

X_train, Y_train, sample_weights = imageFunctions.add_sample_weights(
    X_train, Y_train, 1.0, 3.0
)


Y_train = np.array(Y_train, dtype=np.float32)
Y_test = np.array(Y_test, dtype=np.float32)


iou_scores = []
dice_scores = []

for i in range(50):

    model = create_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model.compile(
        "Adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[sm.metrics.iou_score],
    )
    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou_score", mode="max", patience=50, restore_best_weights=True
    )

    results = model.fit(
        x=X_train,
        y=Y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.15,
        shuffle=True,
        callbacks=[cb],
    )

    acc = results.history["iou_score"]
    val_acc = results.history["val_iou_score"]
    epochs = range(1, len(acc) + 1)

    evaluationFunctions.plotTrainingAcc(acc, val_acc, epochs)

    print("Predicting for test set")
    preds = model.predict(X_test)
    preds_t = preds > 0.5

    evaluationFunctions.plotMasks(preds_t)
    evaluationFunctions.plotMasks(Y_test)

    iou_score = evaluationFunctions.iouScore(preds_t, Y_test)
    dice_score = evaluationFunctions.diceScore(preds_t, Y_test)
    iou_scores.append(iou_score)
    dice_scores.append(dice_score)
    model.save(
        r"C:\Users\joona\Documents\Tohtorikoulu\Uudet ajot\mri_180322\unet_"
        + str(i + 1)
    )


print("Average IoU-score: ", np.average(iou_scores))
print("Average Dice-score: ", np.average(dice_scores))

n = 0
tholds = []
scores = []
for j in range(1000):
    n += 0.001
    temp_preds_t = preds >= n
    d = evaluationFunctions.diceScore(temp_preds_t, Y_test)
    scores.append(d)
    tholds.append(n)
best_thold = tholds[scores.index(max(scores))]
preds_t = preds >= best_thold

loadToExcel("MRI_50x_180322.xlsx", iou_scores, dice_scores)

# med_model = tf.keras.models.load_model(r'C:\Users\joona\Documents\Tohtorikoulu\Uudet ajot\fuusio220222\unet_14', custom_objects={'iou_score': sm.metrics.iou_score})


"""    
dice_scores = []
iou_scores = []
for i in range(len(X_test)): 
    img = np.expand_dims(X_test[i], axis=0)
    pred = model.predict(img)
    pred_t = pred > 0.5
    dsc = evaluationFunctions.diceScore(pred_t, Y_test[i])
    iou = evaluationFunctions.iouScore(pred_t, Y_test[i])
    dice_scores.append(dsc)
    iou_scores.append(iou)
"""
