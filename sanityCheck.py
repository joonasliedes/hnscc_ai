import numpy as np
import matplotlib.pyplot as plt



def checkSanity(all_pos_mri_slices, all_pos_pet_slices, all_mask_slices, all_negative_mri_slices, all_negative_pet_slices, pet_slices, mri_slices, mask_slices): 

    print("#############################SANITY CHECK#####################################")
    print(" pos mri kuvia:{}; pos pet kuvia:{}; maskeja:{}".format(len(all_pos_mri_slices), len(all_pos_pet_slices), len(all_mask_slices)))
    print(" neg mri kuvia:{}; neg pet kuvia:{}".format(len(all_negative_mri_slices), len(all_negative_pet_slices)))
    print("mri muoto:{}; pet muoto:{}; maski muoto:{}".format(all_pos_mri_slices[100].shape, all_pos_pet_slices[300].shape, all_mask_slices[600].shape))
    print("pet kuvat yht:{}; mri kuvat yht: {}; maski yht:{}".format(len(pet_slices),len(mri_slices),len(mask_slices)))
    print(np.min(all_pos_pet_slices), np.max(all_pos_pet_slices))
    print(type(all_pos_pet_slices))
    print(type(all_pos_pet_slices[0]))
    plt.figure(figsize=(10,10))
    for i in range(20,40):
        plt.subplot(4,5,i-19)
        plt.grid(False)
        plt.imshow(mri_slices[i].T)
    plt.show()
    plt.figure(figsize=(10,10))
    for i in range(20,40):
        plt.subplot(4,5,i-19)
        plt.grid(False)
        plt.imshow(pet_slices[i].T)
    plt.show()
    plt.figure(figsize=(10,10))
    for i in range(20,40):
        plt.subplot(4,5,i-19)
        plt.grid(False)
        plt.imshow(mask_slices[i].T)
    plt.show()
    print("#############################SANITY CHECK END#####################################")