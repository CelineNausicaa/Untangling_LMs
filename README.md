# Untangling_LMs

This repository contains one of my personal projects: creating an automatic error checking system using the LLM m-DeBERTa-V3, a brand new model released in 2023 by He et al. (2023)

To run the model you need to:
1. Download the Jupyter Notebook fine_tune_mDeBERTaV3.ipynb
2. Download the data from Spraakbanken: https://github.com/spraakbanken/multiged-2023
3. Obtain the permission from Spraakbanken to obtain the test data
4. Use the eval.py file to compare your predictions with the truth values

I am not allowed to share the data myself, but simply download it from Spraakbanken's repository and ask for the permission to use the labelled test set.

This project is a work in progress. 
My goal for the next few weeks is to improve accuracy/precision/recall/F0.5 scores. There are multiple options that I am exploring:

- Testing different loss functions (colwise MSE, cross-entropy loss)
- Reducing the padding length
- Changing the shape of the tensors
