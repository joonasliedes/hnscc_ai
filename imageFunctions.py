import nibabel as nib
import os
import glob
import numpy as np

rootPath = r'C:\Users\joona\Documents\Tohtorikoulu\Carimas testikuvat'

#Palauttaa listan kuvapakkoja
def getPatients(modality, groundTruth): 
    images = []
    for label in os.listdir(rootPath): 
        if(groundTruth in label): 
            labelPath = os.path.join(rootPath + "\\" + label)
            for patient in os.listdir(labelPath): 
                patientFolderPath = os.path.join(labelPath + "\\" + patient)
                for imgFolder in os.listdir(patientFolderPath):   
                    if(modality in imgFolder): 
                        imgFolderPath = os.path.join(patientFolderPath + "\\" + imgFolder)                    
                        for modalityFolder in os.listdir(imgFolderPath): 
                            modalityFolderPath = os.path.join(imgFolderPath + "\\" + modalityFolder)
                            for img in glob.glob(modalityFolderPath): 
                                if(".img" in img): 
                                    imgPath = os.path.join(img)
                                    img = nib.load(imgPath).get_fdata()
                                    images.append(img)
                                    
    return images
#Palauttaa yksittäiset kuvat listana
#modality = mri/pet/maski
#groundTruth = negatiivinen/positiivinen
def getSlices(patients):
    slices = []
    for i in range(len(patients)):
        mri = patients[i]
        for j in range(mri.shape[2]): 
            slices.append(mri[:,:,j])
    return slices

#Maskileike ei tyhjä --> ota kuva
def getCancerSlices(all_mask_slices, modality_slices): 
    cancer_modality_slices = []
    cancer_mask_slices = []
    for i in range(len(all_mask_slices)): 
        if(np.max(all_mask_slices[i])!=0):
            cancer_modality_slices.append(modality_slices[i])
            cancer_mask_slices.append(all_mask_slices[i])
    
    return cancer_mask_slices, cancer_modality_slices

def stackImages(mri, pet): 
    fused_imgs = []
    for i in range(len(mri)): 
        petMri = np.dstack((mri[i], pet[i]))
        fused_imgs.append(petMri)

    return fused_imgs

def getMinMax(listOfFusedImgs): 
    mri_min = np.min(listOfFusedImgs)
    mri_max = np.max(listOfFusedImgs)

    pet_max = 0
    temp_max = 0
    pet_min = 0
    temp_min = 0
    for i in range(len(listOfFusedImgs)):
        temp_max =  np.max(listOfFusedImgs[i][:,:,1])
        if(pet_max < temp_max):
            pet_max = temp_max
        else: 
            pet_max = pet_max
        temp_min = np.min(listOfFusedImgs[i][:,:,1])
        if(pet_min > temp_min): 
            pet_min = temp_min
        else: 
            pet_min = pet_min

    return mri_min, mri_max, pet_min, pet_max


def normalize(listOfFusedImgs, mri_min, mri_max, pet_min, pet_max):
    
    for i in range(len(listOfFusedImgs)): 
        pet_img = listOfFusedImgs[i][:,:,1]
        mri_img = listOfFusedImgs[i][:,:,0]
        pet_img = (pet_img - pet_min) / (pet_max - pet_min)  
        mri_img = (mri_img - mri_min) / (mri_max - mri_min)
        pet_img = np.clip(pet_img, a_min=0, a_max=1) #make sure all values between 0 and 1
        mri_img = np.clip(mri_img, a_min=0, a_max=1)
        listOfFusedImgs[i][:,:,0] = mri_img
        listOfFusedImgs[i][:,:,1] = pet_img 

def normalizeSingle(listOfIMGs, img_min, img_max):
    
    for i in range(len(listOfIMGs)): 
        img = listOfIMGs[i]
        img = (img - img_min) / (img_max - img_min)
        img = np.clip(img, a_min=0, a_max=1)
        listOfIMGs[i] = img
