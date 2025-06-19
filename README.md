# machine_learning_stock_prediction
Stock prediction using various machine learning techniques.


ENVIRONMENT SETUP

This project uses a Conda environment defined in the finance_env_packages.yaml file to manage dependencies consistently.

To create the environment for the first time, run:

%%
conda env create -f finance_env_packages.yaml
%%

Activate the environment with:

%%
conda activate finance_env
%%

You can verify installed packages by running:

%%
conda list
%%

If you add new packages to the YAML file, update the environment with:

%%
conda env update -n finance_env -f finance_env_packages.yaml --prune
%%

The --prune flag removes packages not listed in the YAML to keep the environment clean.

To remove the environment completely, first deactivate it:

%%
conda deactivate
%%

Then remove it with:

%%
conda env remove -n finance_env
%%