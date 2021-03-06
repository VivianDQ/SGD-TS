> 📋 This is the README.md for code accompanying the following paper

# An Efficient Algorithm For Generalized Linear Bandit: Online Stochastic Gradient Descent and Thompson Sampling

This repository is the official implementation of [An Efficient Algorithm For Generalized Linear Bandit: Online Stochastic Gradient Descent and Thompson Sampling]. 

## Dependencies

To run the code, you will need 
```
Python3, NumPy, scikit-learn, Matplotlib
```
Make sure your ``scikit-learn`` version is above ``0.21.2``. Otherwise, ``sklearn.linear_model.LogisticRegression`` might report error when penalty is set to ``none``.

## Simulation

To get the results of simulation in the paper, run the following command:

```
python3 run_simulation.py
```

## Experiments on Forest cover type data

Before running the experiments, first download the Forest cover type data - ``covtype.data.gz`` from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml//machine-learning-databases/covtype/) and save it inside the ``data`` folder. Do not change the name of the data and there is no need to unzip the data.

The experiments on this dataset have two scenarios:

- Use only the quantitative features and the feature vectors are set to cluster centroids. To get the results, run the following command:

```
python3 covtype.py -d 10
```

- Use both quantitative and categorical features and the feature vectors are randomly chosen from the cluster. To get the results, run the following command:

```
python3 covtype.py -d 55 -center 0
```

## Experiments on Yahoo! Today news artical recommendation data

- Before running the experiments, first get the data. You can find Yahoo! Today data from [Yahoo! Webscope's official website](https://webscope.sandbox.yahoo.com/). Note that getting this data usually needs Webscope's permission.

- After you get the data ``Webscope_R6A.tgz``, please save it inside the ``data`` folder and run the following command to unzip the file. If this is successful, you will be able to see an ``R6`` folder inside the ``data`` folder, and it contains 10 days' data from the Yahoo! Today Module.

```
cd data/
tar zxvf Webscope_R6A.tgz
cd ..
```

- To get the results in the paper, run the following command:

```
python3 yahoo.py
```


## Results

Our implementations will automatically create a ``results`` folder. Numerical results on each data will automatically be saved in separate folders inside ``results`` folder. See the following for details.

- Simulation results are saved in ``results/simulations_d6_k100`` folder.

- For forest cover type data, the averaged regrets of algorithms are saved in ``results/covtype_d10`` and ``results/covtype_d55_rf`` folder respectively. The frequencies of draws for best four arms are save in ``results/covtype_freq_d10`` and ``results/covtype_freq_d55_rf`` folder.

- For Yahoo data, results are saved in ``results/yahoo`` folder.

## Plots

To produce the same plots as in our paper, run the following command, it will create a ``plots`` folder and the figures will be saved there.

```
python3 plot.py
```
