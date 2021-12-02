from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import imageFunctions
import lossFunctions
import random
import augmentationFunctions
from dataToExcel import loadToExcel
import evaluationFunctions
import matplotlib.pyplot as plt
from models import create_unet


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 2
SEED = 1
random.seed(SEED)

#Get patients with cancer
pos_patients_pet = imageFunctions.getPatients("pet","pos")
pos_patients_mri = imageFunctions.getPatients("mri","pos")
pos_patients_masks = imageFunctions.getPatients("mask","pos")

#Get patients without cancer
neg_patients_pet = imageFunctions.getPatients("pet","neg")
neg_patients_mri = imageFunctions.getPatients("mri","neg")

#remove upper 20% of patients image stack 
for i in range(len(neg_patients_mri)):
    limit_mri = round(neg_patients_mri[i].shape[2]*0.8)
    limit_pet = round(neg_patients_pet[i].shape[2]*0.8)
    neg_patients_mri[i] = np.delete(neg_patients_mri[i], np.s_[limit_mri:neg_patients_mri[i].shape[2]],axis=2)
    neg_patients_pet[i] = np.delete(neg_patients_pet[i], np.s_[limit_pet:neg_patients_pet[i].shape[2]],axis=2)
#Get positive slices
all_pos_mri_slices = imageFunctions.getSlices(pos_patients_mri)
all_mask_slices = imageFunctions.getSlices(pos_patients_masks)
all_pos_pet_slices = imageFunctions.getSlices(pos_patients_pet)

#get negative slices
all_negative_mri_slices = imageFunctions.getSlices(neg_patients_mri)
all_negative_pet_slices = imageFunctions.getSlices(neg_patients_pet)


#Choose cancer images with mask from mri slices
cancer_mask_slices, cancer_mri_slices = imageFunctions.getCancerSlices(all_mask_slices, all_pos_mri_slices)

#Choose cancer images with mask from pet slices
cancer_mask_slices, cancer_pet_slices = imageFunctions.getCancerSlices(all_mask_slices, all_pos_pet_slices)

#Choose negative images randomly
negative_mri_slices = []
negative_pet_slices  =[]
for i in range(len(cancer_mri_slices)): 
    rand_numb = random.randrange(0,len(cancer_mri_slices)-1)
    negative_mri_slices.append(all_negative_mri_slices.pop(rand_numb))
    negative_pet_slices.append(all_negative_pet_slices.pop(rand_numb))

#Make negative masks
neg_mask_slices = []
for i in range(len(cancer_mask_slices)): 
    neg_mask_slices.append(np.zeros(shape=cancer_mask_slices[i].shape))

#Merge lists
mri_slices = cancer_mri_slices + negative_mri_slices
pet_slices = cancer_pet_slices + negative_pet_slices
mask_slices = cancer_mask_slices + neg_mask_slices

#Fuse images
fused_images = imageFunctions.stackImages(mri_slices, pet_slices)

stratify_val = [0 if mask.any() else 1 for mask in mask_slices]

train_ids, test_ids, train_labels, test_labels = train_test_split(fused_images, mask_slices, test_size=0.2, random_state=SEED,stratify=stratify_val)



print('Resizing images')

for i in range(len(train_ids)): 
    img = cv2.resize(train_ids[i], (IMG_HEIGHT, IMG_WIDTH),interpolation=cv2.INTER_CUBIC)
    train_ids[i] = img

print('Resizing masks')

for i in range(len(train_labels)): 
    mask = cv2.resize(train_labels[i], (IMG_HEIGHT, IMG_WIDTH),interpolation=cv2.INTER_CUBIC) 
    train_labels[i] = mask

print('Normalising PET and MRI images')
mri_min, mri_max, pet_min, pet_max = imageFunctions.getMinMax(train_ids) #Get min and max values

imageFunctions.normalize(train_ids, mri_min, mri_max, pet_min, pet_max)

X_train = np.array(train_ids, dtype=np.float32)
Y_train = np.array(train_labels, dtype=np.bool)

#X_train = np.expand_dims(X_train, axis=3)
Y_train = np.expand_dims(Y_train, axis=3)

print("DONE")

print('Resizing test images')

for i in range(len(test_ids)): 
    img = cv2.resize(test_ids[i], (IMG_HEIGHT, IMG_WIDTH),interpolation=cv2.INTER_CUBIC)
    test_ids[i] = img
print('resising test masks')
for i in range(len(test_labels)): 
    mask = cv2.resize(test_labels[i], (IMG_HEIGHT, IMG_WIDTH),interpolation=cv2.INTER_CUBIC) 
    test_labels[i] = mask
print('Normalising test images')

imageFunctions.normalize(test_ids, mri_min, mri_max, pet_min, pet_max)

X_test = np.array(test_ids, dtype=np.float32)
Y_test = np.array(test_labels, dtype=np.bool)
Y_test = np.expand_dims(Y_test, axis=3)
print('DONE')

print('Augmenting images')
X_train_aug, Y_train_aug = augmentationFunctions.augment(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS, 5, X_train, Y_train)
print('DONE')

iou_scores = []
dice_scores = []

for i in range(50): 
    
    model = create_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_jaccard_coef',mode='max', patience=50, restore_best_weights=True)

    results = model.fit(x = X_train, y = Y_train, epochs=200, batch_size=32, validation_split=0.1, shuffle=True, callbacks=[cb])

    acc = results.history['jaccard_coef']
    val_acc = results.history['val_jaccard_coef']
    epochs = range(1, len(acc) + 1)

    evaluationFunctions.plotTrainingAcc(acc, val_acc, epochs)

    print('Predicting for test set')
    preds = model.predict(X_test)
    preds_t = preds > 0.5
    print('Done')

    #evaluationFunctions.plotMasks(preds_t) 
    #evaluationFunctions.plotMasks(Y_test)

    iou_score = evaluationFunctions.iouScore(preds_t,Y_test)
    dice_score = evaluationFunctions.diceScore(preds_t,Y_test)
    iou_scores.append(iou_score)
    dice_scores.append(dice_score)
    model.save(r'C:\Users\joona\Documents\Tohtorikoulu\U-net ajot\fuusio_ajot_011221\unet_'+str(i+1))
    

print("Average IoU-score: ", np.average(iou_scores))
print("Average Dice-score: ", np.average(dice_scores))


loadToExcel("Fused_50x_011221.xlsx", iou_scores, dice_scores)

#model = tf.keras.models.load_model(r'C:\Users\joona\Documents\Tohtorikoulu\U-net ajot\unet_2_011121_ilmYO', custom_objects={'jaccard_coef': lossFunctions.jaccard_coef})



'''    
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
'''