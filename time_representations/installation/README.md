# NavGreen  
  
The required packages should be installed using **conda** and **pip**. 

The corresponding requirements are found in **requirements.txt**.

Note that **sklearn** and **pytorch** (*version=2.2.2*) with **cuda** (*version=11.8*) should be installed individually.

Specifically, run:

1.  ```conda create -n <myenv> python=3.10``` To create a new conda environment
    
2.  ```conda activate <myenv>``` To activate the environment
    
3.  ```conda install --file requirements.txt``` To install the requirements
    
4.  ```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia``` To install pytorch and cuda
    
5. ```conda install scikit-learn``` To install sklearn

6. ```pip install astral``` To install astral
  
To train the different models for different seeds run the script : ```hist_data_analysis/train_ml_models.py``` .

To evaluate the models run the script : ```hist_data_analysis/eval_ml_models.py``` .
