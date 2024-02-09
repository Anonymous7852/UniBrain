# UniBrain

Anonymous repository for UniBrain: A Unified Model for End-to-End Brain Imaging Analysis. 
This repository contains the implementation of UniBrain.

### End-to-End Brain Imaging Analysis Problem

![unibrain_problem](./fig/unibrain_problem.png)
The problem of end-to-end brain imaging analysis. Given a set of raw images, each with its corresponding extraction mask and diagnosis label, along with a standard template brain image and its segmentation and parcellation labels, the goal is to train a model to perform brain extraction, registration, segmentation, parcellation, network generation and classification tasks simultaneously.
### Network Architecture
![unibrain_architecture](./fig/unibrain_architecture.png)
An overview of the proposed UniBrain. <span style="color: rgb(181, 181, 181);">*Extraction (Ext)*</span> module removes non-brain tissue of the raw source image **S**, producing the extracted brain image **E**. <span style="color: rgb(251, 153, 140);">*Registration (Reg)*</span> module aligns extracted brain **E** with the target image **T** through affine transformation **A**, resulting in the warped image **W**. <span style="color: rgb(218, 175, 244);">*Segmentation & parcellation (Seg & Parc)*</span> module takes the source image **S**, target segmentation mask **B**, target parcellation mask **P**, and inversed affine transformation **A<sup>-1</sup>** to generate the segmentation mask **R** and the parcellation mask **U**. <span style="color: rgb(249, 203, 156);">*Brain network (BN)*</span> module uses the source image **S** and the parcellation mask **U** to construct brain network *G* by learning ROI feature **H** and connectivity matrix **C**. <span style="color: rgb(255, 127, 80);">*Classification (CLS)*</span> module then uses the ROI feature **H** and connectivity matrix **C** to produce the prediction **ŷ**. All modules are integrated to enable collective learning. UniBrain outputs include the extracted brain image **E**, the target-aligned warped image **W**, the source image's brain tissue segmentation mask **R**, the source image's brain ROI parcellation mask **U**, the brain network connectivity matrix **C** and the prediction **ŷ**.

## File listing

The corresponding files contain the source code and sample data of UniBrain.

+ __dataset__ : Sample dataset for code test
+ __main.py__ : Main code for UniBrain training
+ __model.py__ : Supporting models
+ __preprocess_data.py__ : Using for preprocess neuroimaging data
+ __train.py__ : Supporting training
+ __utils.py__ : Supporting functions

Note that all public datasets used in the paper can be found here:
+ __[ADHD 200](https://fcon_1000.projects.nitrc.org/indi/adhd200/)__ 
+ __[ABIDE](https://fcon_1000.projects.nitrc.org/indi/abide/)__

Raw data can be preprocessed via preprocess_data.py.

## Instructions on training UniBrain

The following script is for training:

```
python main.py
```
<b>Parameters:</b>

+ __main.py__ :

	+ __train_set_name__ : file name of training set, default "adhd_train_sample.npy"
	+ __val_set_name__ : file name of validation set, default "adhd_val_sample.npy"
	+ __test_set_name__ : file name of test set, default "adhd_test_sample.npy"
	+ __enc_nf__ :  extraction network encoder filters numbers, default "[1,16,32,32,64,64,128,128]"
	+ __dec_nf__ :  extraction network decoder filters numbers, default "[128,64,64,32,32,32,16,1]"
    + __enc_affine__ :  registration network encoder filters numbers, default "[2,16,32,64,128,256,512]"
    + __enc_seg__ :  segmentation network encoder filters numbers, default "[1,128,256,256,512,512]"
    + __dec_seg__ :  segmentation network decoder filters numbers, default "[512,256,256,256,128,4]"
    + __enc_par__ :  parcellation network encoder filters numbers, default "[1,128,256,256,512,512]"
    + __dec_par__ :  parcellation network decoder filters numbers, default "[512,256,256,256,128,117]"
    + __feature_net_hidden__ :  feature extraction network hidden numbers, default "256"
    + __feature_net_hidden__ :  feature extraction network output dimensions, default "128"
    + __gnn_hidden__ :  graph network hidden numbers, default "128"
    + __reg_loss_name__ :  training registration similarity loss function , default "GCC"
    + __ext_stage__ :  number of stages of extraction, default "1"
    + __reg_stage__ :  number of stages of registration, default "5"
    + __weight_decay__ :  rate of weight_decay, default "0.000001"
    + __if_train_aug__ :  apply data augmentation during the training, default "True"
    + __if_pred_aal__ :  predict parcellation mask, default "False"
    + __round_num__ :  round number of experiments, default "1"
    + __lambda_ext__ :  extraction loss weight number, default "1"
    + __lambda_reg__ :  registration loss weight number, default "0.1"
    + __lambda_seg__ :  segmentation loss weight number, default "1"
    + __lambda_cls__ :  classification loss weight number, default "1"
    + __batch_size__ : batch size, default 1
    + __img_size__ : size of input images, default 96
    + __num_epochs__ : number of epochs, default 1000
    + __learning_rate__ : learning rate, default 0.00001
    + __save_every_epoch__ : saving interval for results, default 1
    + __save_start_epoch__ : start point for results saving, default 0
    + __model_name__ : model, default UniBrain(img_size, ext_stage , reg_stage, gamma, beta)


## Result
The results can be found after training.
+ __loss_log__ :
    + __model_name.txt__ : log file of the model
+ __model__ :
    + __model_name.pth__ : saved model
+ __sample_img__ :
    + __o__ : target images
    + __t__ : source images
	+ __s_stage__ : extracted images by stage
	+ __s_stage_mask__ : mask of extracted images by stage
    + __r_stage__ : warped (registered) images by stage
    + __segpred_am__ : predicted segmentation mask
    + __reginv_am__ : predicted parcellation mask
    + __adj_m__ : predicted adjacency matrix