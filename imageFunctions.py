import nibabel as nib
import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve

ROOTPATH = r"C:\Users\joona\Documents\Tohtorikoulu\Carimas testikuvat"


def getPatients(modality, groundTruth) -> list:
    images = []
    for label in os.listdir(ROOTPATH):
        if groundTruth in label:
            labelPath = os.path.join(ROOTPATH + "\\" + label)
            for patient in os.listdir(labelPath):
                patientFolderPath = os.path.join(labelPath + "\\" + patient)
                for imgFolder in os.listdir(patientFolderPath):
                    if modality in imgFolder.lower():
                        imgFolderPath = os.path.join(
                            patientFolderPath + "\\" + imgFolder
                        )
                        for img in os.listdir(imgFolderPath):
                            if ".img" in img:
                                imgPath = os.path.join(imgFolderPath + "\\" + img)
                                print(imgPath)
                                img = nib.load(imgPath).get_fdata()
                                images.append(img)
    return images


def getSlices(patients):
    slices = []
    for p in patients:
        for i in range(p.shape[2]):
            slices.append(p[:, :, i, :])

    return slices


def getMaskSlices(patient_masks):
    mask_slices = []
    for p in patient_masks:
        for i in range(p.shape[2]):
            mask_slices.append(p[:, :, i])

    return mask_slices


def getCancerPatients(imgs, labels):
    pos_patients = []
    pos_masks = []
    neg_patients = []
    neg_masks = []
    for i in range(len(imgs)):
        if np.max(labels[i]) > 0:
            pos_patients.append(imgs[i])
            pos_masks.append(labels[i])
        else:
            neg_patients.append(imgs[i])
            neg_masks.append(labels[i])
    return pos_patients, pos_masks, neg_patients, neg_masks


# Maskileike ei tyhjä --> ota kuva
def getCancerSlices(patients, masks):
    img_slices = []
    mask_slices = []
    for i in range(len(patients)):
        for j in range(patients[i].shape[2]):
            if np.max(masks[i][:, :, j]) > 0:
                img_slices.append(patients[i][:, :, j, :])
                mask_slices.append(masks[i][:, :, j])

    return img_slices, mask_slices


def fuseImages(mri, pet):
    fused_imgs = []
    for i in range(len(mri)):
        petMri = np.stack((mri[i], pet[i]), axis=3)
        fused_imgs.append(petMri)

    return fused_imgs


def cutStack(patients, percentage):
    for i in range(len(patients)):
        patients[i] = patients[i][
            :, :, 0 : round((1 - percentage) * patients[i].shape[2]), :
        ]


def getMinMax(listOfFusedImgs):
    mri_min = np.min(listOfFusedImgs)
    mri_max = np.max(listOfFusedImgs)

    pet_max = 0
    temp_max = 0
    pet_min = 0
    temp_min = 0
    for i in range(len(listOfFusedImgs)):
        temp_max = np.max(listOfFusedImgs[i][:, :, 1])
        if pet_max < temp_max:
            pet_max = temp_max
        else:
            pet_max = pet_max
        temp_min = np.min(listOfFusedImgs[i][:, :, 1])
        if pet_min > temp_min:
            pet_min = temp_min
        else:
            pet_min = pet_min

    return mri_min, mri_max, pet_min, pet_max


def normalize(listOfFusedImgs, mri_min, mri_max, pet_min, pet_max):

    for i in range(len(listOfFusedImgs)):
        pet_img = listOfFusedImgs[i][:, :, 1]
        mri_img = listOfFusedImgs[i][:, :, 0]
        pet_img = (pet_img - pet_min) / (pet_max - pet_min)
        mri_img = (mri_img - mri_min) / (mri_max - mri_min)
        pet_img = np.clip(
            pet_img, a_min=0, a_max=1
        )  # make sure all values between 0 and 1
        mri_img = np.clip(mri_img, a_min=0, a_max=1)
        listOfFusedImgs[i][:, :, 0] = mri_img
        listOfFusedImgs[i][:, :, 1] = pet_img


def normalizeSingle(listOfIMGs, img_min, img_max):

    for i in range(len(listOfIMGs)):
        img = listOfIMGs[i]
        img = (img - img_min) / (img_max - img_min)
        img = np.clip(img, a_min=0, a_max=1)
        listOfIMGs[i] = img


def add_sample_weights(image, label, w1, w2):
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant([w1, w2])
    class_weights = class_weights / tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


# from https://github.com/hmhellstrom/ai-cancer-project/blob/python/main.py
def compute_threshold(true_labels, predicted):
    fpr, tpr, thresholds = roc_curve(true_labels, predicted, drop_intermediate=False)
    J_stats = tpr - fpr
    opt_thresholds = thresholds[np.argmax(J_stats)]
    return opt_thresholds
