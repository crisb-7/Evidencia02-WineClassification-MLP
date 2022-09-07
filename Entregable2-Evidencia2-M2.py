#!/usr/bin/env python
# coding: utf-8

# # Importing modules

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, classification_report


# # Import Data Frame

# In[2]:


columnsWine = ["WineID", "Alcohol", "MalicAcid", "Ash", "AlcalinityOfAsh", "Magnesium", "TotalPhenols", 
               "Flavanoids", "NonflavanoidPhenols","Proanthocyanins", "ColorIntensity", "Hue", "OD280/OD315", "Proline"]

df = pd.read_csv(r"C:\Users\Cristofer Becerra S\Documents\Tec de Monterrey\7mo Semestre\Datasets\wine.data", 
                 names = columnsWine)


# In[3]:


df.head()


# # Preprocessing data for the Neural Network

# We wish to standardize the input of the neural network to align with the Multi-layer Perceptron Classifier's output. We apply a Min-Max scaling to our feature data and a Label Encoder to our wine categories.

# In[4]:


le = LabelEncoder()
# scaler = StandardScaler()
minmax = MinMaxScaler()


# In[5]:


x = minmax.fit_transform(df.drop(columns = "WineID"))
y = le.fit_transform(df.WineID)


# # Train-validate-test split

# We divide the dataset into 3 subsets: **train** (56%), **validate** (24%), and **test** (20%). We will also work with the same random state (0) for all Sklearn's functions for reproducibility.

# In[6]:


rs, rsData = 0, 0
x1, x_test, y1, y_test = train_test_split(x, y, train_size = 0.8, random_state = rsData)
x_train, x_validate, y_train, y_validate = train_test_split(x1, y1, train_size = 0.7, random_state = rsData)


# # Neural Network Comparison

# We will compare the performance of two neural networks with a fixed set of epochs (50). The goal is to obtain the best score without overfitting witht the given restraint (epochs).

# In[7]:


epochs = 50


# ## Simple NN
# 
# First we build a simple neural network. Let a simple neural network have 1 hidden layer with 2 neurons, optimized by *optimized* Stochastic Gradient Descent (Adaptive Moment Estimation, Adam) using a ReLu activation function, an adaptive learning rate starting at 0.01.

# In[8]:


simpleNN = MLPClassifier(random_state = rs, hidden_layer_sizes = (2,), learning_rate = "adaptive", 
                         learning_rate_init = 0.01, max_iter = epochs)


# We can train and overview its performance by a simple mean accuracy score for both training and test datasets.

# In[9]:


simpleNN.fit(x_train, y_train)
print("Training score: ", round(simpleNN.score(x_train, y_train), 4))
print("Validate score: ", round(simpleNN.score(x_validate, y_validate), 4))
print("Test score: ", round(simpleNN.score(x_test, y_test), 4))


# It appears that the optimization does not converge with the iterations constraint, but it yields decent results without overfitting. Given more iterations the loss function will indeed minimize and yield better scores, and hopefully it doesn't fall into overfitting territory.

# In[10]:


del simpleNN


# ## Optimized NN

# Now, we can perform a *rudimentary* optimization of the parameters of the network. This can be done using Scikit-learn's GridSearch algorithm for exhaustive search of the best model using different combinations passed to it.

# ### Simple Grid Search Optimization

# In[11]:


from sklearn.model_selection import GridSearchCV


# Since Adam is an optimizer of the simple SGD, we can reduce the combinations by only employing this version of SGD. Also, given the functionality of Scikit-learn and the fact that adaptive learning rate is more practical than constant (generally yielding better results for rudimentary implementations such as this one), we will also reduce the parameters by only using adaptive learning rates. We will focus the grid search on the structure of the layers, the learning rate and the regularization parameter.

# In[12]:


hyperparameters = {'activation': ['relu'], 
                   'hidden_layer_sizes': [(2,), (4,), (8,), (16,), (64,), 
                                          (2,2,), (4,2), (2,4), (8,2), (2,8), (8,4), (4,8),
                                         (2,2,2), (4,2,2), (2,4,2), (4,2,4), (4,8,2), (4,8,4), (8,8,8)],
                   'alpha': [0.0001, 0.0005, 0.001, 0.01],
                   'learning_rate_init': [0.001, 0.01, 0.1],
                   'learning_rate': ['adaptive'],
                   'solver': ['adam']}


# **DANGER**: this is an intensive CPU cell. Do not run this cell if your computer isn't very capable or if you're running it in Google Colab (it'll truly take forever). You've been warned.

# In[13]:


nn = MLPClassifier(random_state = rs, max_iter = epochs)

hps = GridSearchCV(nn, hyperparameters, n_jobs = -1, cv = 5) # hyper-parameter search

hps.fit(x_train, y_train);

bestH = hps.best_params_

SearchResults = pd.DataFrame(hps.cv_results_)
SearchResults = SearchResults.sort_values("rank_test_score", ascending = True)
SearchResults.head()


# ## Performance Comparison

# Once we have obtained our "optimized" parameters, we proceed to define them in order to compare their performance.

# In[14]:


simpleNN = MLPClassifier(random_state = rs, hidden_layer_sizes = (2,), learning_rate = "adaptive", 
                         learning_rate_init = 0.01, max_iter = epochs)

# Best model so far - 1.0 train, validate, test; convergence in 110 iterations
# optimizedNN = MLPClassifier(random_state = rs, hidden_layer_sizes = (8,8,16), learning_rate = "adaptive", 
#                             alpha = 0.01, learning_rate_init = 0.01, max_iter = 110)

optimizedNN = MLPClassifier(random_state = rs, hidden_layer_sizes = (8,8,8), learning_rate = "adaptive", 
                            alpha = 0.001, learning_rate_init = 0.01, max_iter = epochs)


# We run a partial fit with the set number of epochs and storing their respective scores each iteration.

# In[15]:


simpleTrainScore, simpleValidateScore, optimizedTrainScore, optimizedValidateScore = [], [], [], []
simpleTestScore, optimizedTestScore = [], []

for i in range(0, epochs):
    
    simpleNN.partial_fit(x_train, y_train, np.unique(y_train))
    
    simpleTrainScore.append(simpleNN.score(x_train, y_train))
    simpleValidateScore.append(simpleNN.score(x_validate, y_validate))
    simpleTestScore.append(simpleNN.score(x_test, y_test))
    
    optimizedNN.partial_fit(x_train, y_train, np.unique(y_train))
    
    optimizedTrainScore.append(optimizedNN.score(x_train, y_train))
    optimizedValidateScore.append(optimizedNN.score(x_validate, y_validate))
    optimizedTestScore.append(optimizedNN.score(x_test, y_test))


# Peeking into the overall performance of our "optimized" NN:

# In[16]:


print("Training score: ", optimizedNN.score(x_train, y_train))
print("Validate score: ", optimizedNN.score(x_validate, y_validate))
print("Testing score: ", optimizedNN.score(x_test, y_test))


# We can see that the NN performs extremely well in comparison with the simple NN, and is also not in overfitting territory; on the contrary, the NN performs better in the validate and test sets.

# ## Accuracy & Loss

# Plotting the evolution of the accuracy on all sets and the loss function

# In[17]:


fig, axes = plt.subplots(1,2, figsize = (12,4))

axes[0].plot(simpleTrainScore, label = "Train - Simple ")
axes[0].plot(simpleValidateScore, label = "Validate - Simple")
axes[0].plot(simpleTestScore, label = "Test - Simple")

axes[0].plot(optimizedTrainScore, label = "Train - Optimized ")
axes[0].plot(optimizedValidateScore, label = "Validate - Optimized")
axes[0].plot(optimizedTestScore, label = "Test - Optimized")

axes[0].set_title("Accuracy Evolution")
axes[0].set_xlabel(r"Number of iterations, $n$")
axes[0].set_ylabel("Mean Accuracy")
axes[0].set_ylim([0.1, 1.05])
#axes[0].set_xlim([20, 80])
axes[0].grid(True)
axes[0].legend()

axes[1].plot(simpleNN.loss_curve_, label = "Simple")
axes[1].plot(optimizedNN.loss_curve_, label = "Optimizada")
axes[1].set_title("Loss Function")
axes[1].set_xlabel(r"Number of iterations, $n$")
axes[1].set_ylabel(r"$J(\Theta)$")
axes[1].legend()
axes[1].grid(True)

plt.show()


# ## Classification Metrics

# We can also inspect the metrics of classification with a prediction on the validating set

# In[18]:


print(classification_report(y_validate, simpleNN.predict(x_validate), target_names=["Wine A", "Wine B", "Wine C"]))


# In[19]:


print(classification_report(y_validate, optimizedNN.predict(x_validate), target_names=["Wine A", "Wine B", "Wine C"]))


# ## Confusion Matrices

# Plotting the confusion matrices for both neural networks in the 3 subsets

# In[20]:


fig, axes = plt.subplots(2, 3, figsize = (12, 6))

sns.heatmap(confusion_matrix(y_train, simpleNN.predict(x_train)), annot = True, cmap = "Purples", ax = axes[0,0])
axes[0,0].set_title("Train - Simple")

sns.heatmap(confusion_matrix(y_validate, simpleNN.predict(x_validate)), annot = True, cmap = "Purples", ax = axes[0,1])
axes[0,1].set_title("Validate - Simple")

sns.heatmap(confusion_matrix(y_test, simpleNN.predict(x_test)), annot = True, cmap = "Purples", ax = axes[0,2])
axes[0,2].set_title("Test - Simple")

sns.heatmap(confusion_matrix(y_train, optimizedNN.predict(x_train)), annot = True, cmap = "Blues", ax = axes[1,0])
axes[1,0].set_title("Train - Optimized")

sns.heatmap(confusion_matrix(y_validate, optimizedNN.predict(x_validate)), annot = True, cmap = "Blues", ax = axes[1,1])
axes[1,1].set_title("Validate - Optimized")

sns.heatmap(confusion_matrix(y_test, optimizedNN.predict(x_test)), annot = True, cmap = "Blues", ax = axes[1,2])
axes[1,2].set_title("Test - Optimized")

plt.tight_layout()
plt.show()


# ## Performance as a function of Train-Test split

# We can inspect the beahavior of our "optimized" NN as a train-test split changes proportions.

# In[21]:


train_score, test_score = [], []

PS = np.linspace(0.01, 0.99, epochs)

for ps in PS:
    
    x1, x_test, y1, y_test = train_test_split(x, y, train_size = ps, random_state = rsData)
    
    optimizedNN.fit(x_train, y_train);
    
    train_score.append(optimizedNN.score(x_train, y_train))
    test_score.append(optimizedNN.score(x_test, y_test))


# In[22]:


print("Training score: ", np.array(train_score).mean())
print("Testing score: ", np.array(test_score).mean())


# The mean score for both subsets suggests that the NN is very stable with different proportions of training and testing data. We can plot this variation as a function of the split percentage.

# In[23]:


plt.plot(PS, train_score, label = "Train - Optimized ")
plt.plot(PS, test_score, label = "Test - Optimized")
plt.title("Train-Test Split %")
plt.ylim([0.95, 1.02])
plt.grid(True)
plt.legend()
plt.show()


# ## Performance as a function of Random State

# Another important factor that can impact performance is the random state of the data split. We can test this in a similar fashion as the last evolution (train-test split)

# In[24]:


train_score_rs, test_score_rs = [], []

for i in range(epochs):
    
    x1, x_test, y1, y_test = train_test_split(x, y, train_size = 0.7, random_state = i)
    
    optimizedNN.fit(x_train, y_train);
    
    train_score_rs.append(optimizedNN.score(x_train, y_train))
    test_score_rs.append(optimizedNN.score(x_test, y_test))


# In[25]:


print("Training score: ", np.array(train_score_rs).mean())
print("Testing score: ", np.array(test_score_rs).mean())


# Once again, the NN performs extremely well in a very stable manner. We can also plot the evolution

# In[26]:


plt.plot(train_score_rs, label = "Train - Optimized ")
plt.plot(test_score_rs, label = "Test - Optimized")
plt.title("Random State")
plt.ylim([0.9, 1.02])
plt.grid(True)
plt.legend()
plt.show()


# In this case we can see that testing performance oscillates between 0.98 and 1. This indicates that our NN is sensible to overfitting with different random samples of the data. This makes sense since the optimization with grid search was performed with a fixed random state of data.

# # Model Predictions

# Finally, we can make specific predictions with our model. We extract the first five observations of the test dataset and make a Pandas DataFrame for visualization purposes.

# In[27]:


x_predictions = pd.DataFrame(x_test[0:5], columns = columnsWine[1:])
x_predictions.head()


# Then we can create another Data Frame to store the real category of wine, and then add the column of the predictions yielded by our optimized neural network.

# In[28]:


y_prediction = pd.DataFrame(y_test[0:5], columns = ["Real"])
y_prediction["Prediction"] = optimizedNN.predict(x_test[0:5])
y_prediction


# We get an expected result: our neural network aced the five predictions.
