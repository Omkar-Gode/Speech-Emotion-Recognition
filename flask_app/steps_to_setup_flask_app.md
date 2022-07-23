# Steps to make flask app 

## Step 1 : 
### Create Folder in computer (project folder)

## Step 2 : 
### Navigate to it by using cmd

## Step 3 : 
### create virtual environment using " virtualenv venv_folder_name " command

## Step 4 : 
### activate it using " venv_folder_name\Scripts\activate " command

## Step 5:
### install python libraries using following commands in virtual environment:
#### pip install flask
#### pip install numpy
#### pip install pandas
#### pip install tensorflow
#### pip install librosa
#### pip install pickle

## Step 6:
### copy and paste "static", "template" folders and "app.py" file to project folder


## Step 7:
### make a new folder named "model_files" in project folder and put label encoder file, scaling file and model file inside "model_files" named as "Enc_labels.csv", "Scaling_Parameters_copy" and "SER_better_model.h5" respectively


## Step 8:
### finally, you can run app using command "flask run" command

for stopping app press "Ctrl+C"
for deactivating venv "venv_folder_name\Scripts\deactivate"
