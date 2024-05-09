# NavGreen  
  
The required packages should be installed using **conda** and **pip**. 
The corresponding requirements are found in **requirements.txt**.
Note that **sklearn** and **pytorch** (*version=2.2.2*) with **cuda** (*version=11.8*) should be installed individually.

Specifically, run:

```conda create -n <myenv> python=3.10``` To create a new conda environment
```conda activate <myenv>``` To activate the environment
```conda install --file requirements.txt``` To install the requirements
```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia``` To install pytorch and cuda
```conda install scikit-learn``` To install sklearn
  
To train the different models for different seeds run the script : ```hist_data_analysis/train_ml_models.py``` .
To evaluate the models run the script : ```hist_data_analysis/eval_ml_models.py``` .
