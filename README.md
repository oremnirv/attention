# attention

This project tries to infer values of one sequence (A) from 
some points on (A) and other points from another sequence (B). 
The idea is to learn the relationship (the grammar) between pairs of sequences that share some commonality (kernel).   
For this I'm using an attention based model inspired by the Transformer model. 

### The problem 
After training a model and getting positive results in terms 
of metrics like MSE and R squared and eyeballing, the inference step produces typically plots like the one below:

![](infer_bug)

* It is common to observe an immediate and sharp drop from the red dots to the first predictions (golden or blue)
* It is common to observe the predictions matching (more or less) the desired shape after the sudden drop
* the black colored sequence is seen by our network, plus the red dots
* the red line is the observed sequence we would like to match
* the golden points are the mean predictions of our network 
* the blue dots are sampled from N(mean, var) given by our network

### The math

A simplified version of the math in the network can be found ![here](model_maths.pdf).
Although it is not comprehensive, it contains the salient features of this network.

### Setup: 
** this setup works for mac and might require adjustments otherwise
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
   The inference setup is that our network gets to "see" one full sequence (from the pair)
   and only the beginning of the other sequence (see implementation in helpers/plotter.py --> infer_plot2D). Now, our task is to infer the rest of the unseen 
   sequence. 
   
   - Go to last cell: 
        - set parameters:
          ```angular2html
            samples := # times to make inferences
            context_p := # points you want to see at the beginning of the sequence to be inferred
            consec := should the context points be chosen without gaps? (TRUE) or with? (False)
            order := should the data points be ordered before the network "sees" them?    
            num_steps := 999 if we wish to infer the full sequence, else choose an integer in [1, 200 - context_p]
          ```
   - Run the cell. Its output will produce: 
     * a plot where the black line is given to the network, plus the extra red points. 
     The goal is to predict the continuation of the red line. The golden points are the mean prediction
     and the light blue points are samples.
     * The many plots underneath the bigger plot, show the attention the network
    has chosen to give (orange dots) when making a prediction for the green point. 
       

4. If you would like to see some statistics about the weights, loss and embeddings run from terminal:
```angular2html
tensorboard --logdir ~/Downloads/GPT_rbf_const_2D/logs/
``` 
Then go to ```http://localhost:6006/ ``` and select only run_7.

** When looking at the scalars on tensorboard, set the offset to relative

5. Would you like to see any layer in T-SNE? 
    - go to the penultimate cell (after you either trained or loaded a pretrained model)
    - If you want to see the embedding layer, just run as-is and go to step (4), otherwise
    you would have to specify the layer number you are interested in and its metadata. 