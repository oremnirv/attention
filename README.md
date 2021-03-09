# attention

### Setup: 
1. Create a new virtual env (!make sure it is python 3.7 version)
```angular2html
conda create -n debug_omer anaconda python=3.7
```

2. Activate the env. 
```angular2html
conda activate debug_omer
```

3. Clone my GitHub repo. and checkout to omer branch
```angular2html
git clone https://github.com/oremnirv/attention.git
cd attention
git checkout omer

# If you want to push stuff please create your own branch!

```

4. Install required packages 
```angular2html
cd ..
chmod +x attention/local_setup/setup1.sh
./attention/local_setup/setup1.sh
```

5. Download data from ![link](https://universityofcambridgecloud-my.sharepoint.com/:f:/g/personal/on234_cam_ac_uk/ElcsER18Eb1OlZpzjihaAUkB_hiJVxsDuhyaxITNsHq2YQ?e=uKCgyz) and place it in your Downloads folder


6. Now you are ready! run the following
```angular2html
cd attention 
cd notebooks
jupyter notebook UNN.ipynb

```

### Running the notebook:

1. The second cell will prompt you with three questions. 
If you don't wan to create new data just answer:
```
rbf_const_2D  # this means we are using rbf kernel with a constant shift
1 # (how many data points)
False # (overwrite data?)   
```

2. Do you want to train the model? 
    #### a. loading a pretrained model: 
      
   The current trained model is shuffled at each batch and then the first 50
   points are used as context points (observed) to infer the rest of the sequence. 
      - go to cell starting with "if __name__=='__main__"
      - make sure the run parameter is set to 7 (runs 1-6 are also available but did not train well. If you want to know what they are about, please check the recent github commits)
      - choose a training regime by setting the ```tr_regime``` parameter ('shuffle', 'full infer', ' half half') - for details about the different regimes please see batch_creator.py in the data folder
        The current trained model used the ```shuffle``` regime. (I'll add more trained runs once they are performing well)
        
    #### b. Training from scratch:
   
      - Go to cell starting with "if __name__=='__main__"
      - The first two rows after ``tf.random.set.seed`` have parameters you can play with.
        Make sure the run parameter does not already exist (check it in ```/Downloads/GPT_rbf_const_2D/ckpt```).
        Other parameters: 
        ```
        e := embedding dim
        heads := how many separate attention dimensions would you like to have. (see model/dot_prod_attention.py)
        l := set the dimension of layers in the network, coming after the attention. (see model/experimental2d_model.py)
        context := how many points from each pair of sequnces will serve as observed (only relevant if tr_regime is 'shuffle' or 'half half')
        ```
        

3. Do you want to make inferences? 
