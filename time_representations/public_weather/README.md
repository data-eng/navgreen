Here's the corrected version of the text in .md format:

# NavGreen  

## RESULTS USED IN THE PAPER 

In the folder `./hist_data_analysis/models`, you will find the results of the experiments for the three (3) different classification tasks and six (6) different initialization seeds that correspond to training parameters.

Regarding the classification tasks, the folder `FixedBin` corresponds to 'Balanced Ranges', `ValueCountBin` to 'Balanced Classes', and `ValueRangeBin` to 'Max Margins', as referred to in the paper.

Each classification task folder includes the results for each model architecture: `interpolation` being the RNN, `mTAN` the mTAN, and `transformer` the CycTime, as referred to in the paper.

Within each model folder, you will find the runs with each seed, including:

1. `data.csv`: Aggregated testing data, with true and predicted values as well as probabilities.
2. `*.pth`: The trained model.
3. `*.png`: Different metrics illustrated for evaluation purposes.
4. `*_checkpoints.json`: Training (and validation) and testing records.
5. `train_losses.json`: Documentation of the training (and validation) loss per epoch.

From all the seeds, we selected the best model to be the one with the minimal validation set loss.

The model folder also contains the accumulated and aggregated testing and training results in JSON format: `acc_*_stats.json`.

## RUN THE CODE 

To replicate the results by running the training and/or inference, follow these instructions.

**Environment Initialization**

The required packages should be installed using **conda** and **pip**.

The corresponding requirements are found in **requirements.txt**.

Note that **sklearn** and **pytorch** (version=2.2.2) with **cuda** (version=11.8) should be installed individually.

Specifically, run:

1. `conda create -n <myenv> python=3.10` to create a new conda environment.
    
2. `conda activate <myenv>` to activate the environment.
    
3. `conda install --file requirements.txt` to install the requirements.
    
4. `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` to install PyTorch and CUDA.
    
5. `conda install scikit-learn` to install sklearn.

**Run training and/or inference scripts**

To train the different models for different seeds, run the script: `hist_data_analysis/train_ml_models.py`.

To evaluate the models, run the script: `hist_data_analysis/eval_ml_models.py`.

Please note that the data should be stored in a parent folder called `/data`.
