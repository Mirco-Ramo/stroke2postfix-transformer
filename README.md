# Stroke2Postfix Transformer

## Setup

### Linux shell
````
git clone https://csgitlab.ucd.ie/Mirco-Ramo/stroke2postfix-transformer.git
cd './stroke2postfix-transformer'
pip install -r requirements.txt
````

### Python
````
!git clone https://csgitlab.ucd.ie/Mirco-Ramo/stroke2postfix-transformer.git
%cd './stroke2postfix-transformer'
!pip install -r requirements.txt
````

The clone operation could take up to 5 minutes, since all cache files are also downloaded.

## Training and evaluating
The models/notebooks folder exposes a train and a test notebook corresponding to each version presented in the paper, called respectively vX_train.ipynb and vX_test.ipynb, where X is a placeholder for the desired version.

### Data
It is highly recommended to use the cached versions of the database, since the creation of a new one could require a considerable amount of time.

If you decide to regenerate it (discouraged), you just need to empty the cache directory and set ````use_cache```` to ````False````.

````python
d_gen = SequenceGenerator(
        vocab = VOCAB,
        allow_brackets = True,
        save_mode = 'marked_postfix',      # saves expressions in RPN
        total_expressions = 120 * 1000  #final number of expressions is this*augmentation amount
      )
use_cache = False

if use_cache: # Generate from cache file
    train, valid, test = d_gen.generate_from_cache()

else: # Generate from scratch and cache (if regenerated, results could change)
    train, valid, test = d_gen.generate()
````

### Training

Every model v1-v11 is provided already trained: their respective checkpoints are in the check_points folder.

To re-train any of the proposed models, all you have to do is to re-run completely the train notebook of it.

Note that the training phase can be stopped at any time without errors.

You can also decide to experiment new hyperparameters or combinations: just pass them to the ````model_builder```` function when you call it.
As a result, a new json file is saved in the hyperparameters directory and the training check-points of your models will be tracked into the check_points directory
````python
model= model_builder(name=VERSION, vocab=VOCAB, enc_heads=2, dec_pf_dim=128)
````
you can also easily transfer-learn any other encoder or decoder of other models: they will be initialized with the weights saved in check_points

````python
model= model_builder(name=VERSION, vocab=VOCAB, encoder="v11", decoder="v2")
````

WARNING: if you decide to train a new model, REMEMBER TO CHANGE THE NAME YOU PASS TO THE ````model_builder````, OTHERWISE YOU WILL OVERWRITE THE OLD ONE.

You can also resume the training of any model from the point in which it was stopped, just set ````resume=True```` in the call to model's ````train_loop````.
The training will be resumed using the same LR, Optimizer and Scheduler it had when stopped/completed. You can change that passing your custom values to the ````train_loop````

````python
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8))
model.train_loop(resume=True,
                 train_set=train_set,
                 valid_set=valid_set,
                 optimizer=optimizer,
                 scheduler=scheduler
````

### Evaluation
To evaluate any of the proposed model, all you have to do is to execute the relative test notebook.
You don't have to re-run everything, just the tests you are interested in, every test section specifies the other sections it need to be run in order for it to work
(e.g., Section 2 requires sections 0 and 1). This specification is reported in the section title.

If you want to evaluate your custom model, just declare the name of it (in the ````VERSION```` field) and run the test you are interested in, always respecting specified section dependencies.

## References
If you embed any part of this code in your publication, please reference to:

PUT REF HERE