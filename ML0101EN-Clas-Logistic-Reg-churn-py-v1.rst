
.. raw:: html

   <h1 align="center">

 Logistic Regression with Python

.. raw:: html

   </h1>

In this notebook, you will learn Logistic Regression, and then, you'll
create a model for a telecommunication company, to predict when its
customers will leave for a competitor, so that they can take some action
to retain the customers.

 ## What is the difference between Linear and Logistic Regression?

While Linear Regression is suited for estimating continuous values (e.g.
estimating house price), it is not the best tool for predicting the
class of an observed data point. In order to estimate the class of a
data point, we need some sort of guidance on what would be the **most
probable class** for that data point. For this, we use **Logistic
Regression**.

.. raw:: html

   <div class="alert alert-success alertsuccess"
   style="margin-top: 20px">

Recall linear regression: As you know, **Linear regression** finds a
function that relates a continuous dependent variable, *y*, to some
predictors (independent variables *x1*, *x2*, etc.). For example, Simple
linear regression assumes a function of the form:

.. math::


   y = 𝜃0 + 𝜃1 * x1 + 𝜃2 * x2 +...

 and finds the values of parameters *θ0*, *θ1*, *𝜃2*, etc, where the
term *𝜃0* is the "intercept". It can be generally shown as:

.. math::


   ℎ_θ(𝑥) = 𝜃^TX

.. raw:: html

   <p>

.. raw:: html

   </p>

.. raw:: html

   </div>

Logistic Regression is a variation of Linear Regression, useful when the
observed dependent variable, *y*, is categorical. It produces a formula
that predicts the probability of the class label as a function of the
independent variables.

Logistic regression fits a special s-shaped curve by taking the linear
regression and transforming the numeric estimate into a probability with
the following function, which is called sigmoid function 𝜎:

.. math::


   ℎ_θ(𝑥) = 𝜎({θ^TX}) =  \frac {e^{(θ0 + θ1 * x1 + θ2 * x2 +...)}}{1 + e^{(θ0 + θ1 * x1 + θ2 * x2 +...)}}

 Or:

.. math::


   ProbabilityOfaClass_1 =  P(Y=1|X) = 𝜎({θ^TX}) = \frac{e^{θ^TX}}{1+e^{θ^TX}} 

In this equation, :math:`{θ^TX}` is the regression result (the sum of
the variables weighted by the coefficients), ``exp`` is the exponential
function and :math:`𝜎(θ^TX)` is the sigmoid or `logistic
function <http://en.wikipedia.org/wiki/Logistic_function>`__, also
called logistic curve. It is a common "S" shape (sigmoid curve).

So, briefly, Logistic Regression passes the input through the
logistic/sigmoid but then treats the result as a probability:

The objective of **Logistic Regression** algorithm, is to find the best
parameters θ, for :math:`ℎ_θ(𝑥)` = 𝜎(\ :math:`{θ^TX}`), in such a way
that the model best predicts the class of each case.

Customer churn with Logistic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A telecommunications company is concerned about the number of customers
leaving their land-line business for cable competitors. They need to
understand who is leaving. Imagine that you’re an analyst at this
company and you have to find out who is leaving and why.

Lets first import required libraries:

.. code:: ipython3

    import pandas as pd
    import pylab as pl
    import numpy as np
    import scipy.optimize as opt
    from sklearn import preprocessing
    %matplotlib inline 
    import matplotlib.pyplot as plt

About dataset
~~~~~~~~~~~~~

We’ll use a telecommunications dataset for predicting customer churn.
This is a historical customer dataset where each row represents one
customer. The data is relatively easy to understand, and you may uncover
insights you can use immediately. Typically it’s less expensive to keep
customers than acquire new ones, so the focus of this analysis is to
predict the customers who will stay with the company.

This data set provides information to help you predict what behavior
will help you to retain customers. You can analyze all relevant customer
data and develop focused customer retention programs.

The dataset includes information about:

-  Customers who left within the last month – the column is called Churn
-  Services that each customer has signed up for – phone, multiple
   lines, internet, online security, online backup, device protection,
   tech support, and streaming TV and movies
-  Customer account information – how long they’ve been a customer,
   contract, payment method, paperless billing, monthly charges, and
   total charges
-  Demographic info about customers – gender, age range, and if they
   have partners and dependents

Load the Telco Churn data
~~~~~~~~~~~~~~~~~~~~~~~~~

Telco Churn is a hypothetical data file that concerns a
telecommunications company's efforts to reduce turnover in its customer
base. Each case corresponds to a separate customer and it records
various demographic and service usage information. Before you can work
with the data, you must use the URL to get the ChurnData.csv.

To download the data, we will use ``!wget`` to download it from IBM
Object Storage.

.. code:: ipython3

    #Click here and press Shift+Enter
    !wget -O ChurnData.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv

**Did you know?** When it comes to Machine Learning, you will likely be
working with large datasets. As a business, where can you host your
data? IBM is offering a unique opportunity for businesses, with 10 Tb of
IBM Cloud Object Storage: `Sign up now for
free <http://cocl.us/ML0101EN-IBM-Offer-CC>`__

Load Data From CSV File
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    churn_df = pd.read_csv("ChurnData.csv")
    churn_df.head()

Data pre-processing and selection
---------------------------------

Lets select some features for the modeling. Also we change the target
data type to be integer, as it is a requirement by the skitlearn
algorithm:

.. code:: ipython3

    churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
    churn_df['churn'] = churn_df['churn'].astype('int')
    churn_df.head()

Practice
--------

How many rows and columns are in this dataset in total? What are the
name of columns?

.. code:: ipython3

    # write your code here
    churn_df.shape
    


Lets define X, and y for our dataset:

.. code:: ipython3

    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    X[0:5]

.. code:: ipython3

    y = np.asarray(churn_df['churn'])
    y [0:5]

Also, we normalize the dataset:

.. code:: ipython3

    from sklearn import preprocessing
    X = preprocessing.StandardScaler().fit(X).transform(X)
    X[0:5]

Train/Test dataset
------------------

Okay, we split our dataset into train and test set:

.. code:: ipython3

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)

Modeling (Logistic Regression with Scikit-learn)
================================================

Lets build our model using **LogisticRegression** from Scikit-learn
package. This function implements logistic regression and can use
different numerical optimizers to find parameters, including
‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. You can find
extensive information about the pros and cons of these optimizers if you
search it in internet.

The version of Logistic Regression in Scikit-learn, support
regularization. Regularization is a technique used to solve the
overfitting problem in machine learning models. **C** parameter
indicates **inverse of regularization strength** which must be a
positive float. Smaller values specify stronger regularization. Now lets
fit our model with train set:

.. code:: ipython3

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    LR

Now we can predict using our test set:

.. code:: ipython3

    yhat = LR.predict(X_test)
    yhat

**predict\_proba** returns estimates for all classes, ordered by the
label of classes. So, the first column is the probability of class 1,
P(Y=1\|X), and second column is probability of class 0, P(Y=0\|X):

.. code:: ipython3

    yhat_prob = LR.predict_proba(X_test)
    yhat_prob

Evaluation
----------

jaccard index
~~~~~~~~~~~~~

Lets try jaccard index for accuracy evaluation. we can define jaccard as
the size of the intersection divided by the size of the union of two
label sets. If the entire set of predicted labels for a sample strictly
match with the true set of labels, then the subset accuracy is 1.0;
otherwise it is 0.0.

.. code:: ipython3

    from sklearn.metrics import jaccard_similarity_score
    jaccard_similarity_score(y_test, yhat)

confusion matrix
~~~~~~~~~~~~~~~~

Another way of looking at accuracy of classifier is to look at
**confusion matrix**.

.. code:: ipython3

    from sklearn.metrics import classification_report, confusion_matrix
    import itertools
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    print(confusion_matrix(y_test, yhat, labels=[1,0]))

.. code:: ipython3

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
    np.set_printoptions(precision=2)
    
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

Look at first row. The first row is for customers whose actual churn
value in test set is 1. As you can calculate, out of 40 customers, the
churn value of 15 of them is 1. And out of these 15, the classifier
correctly predicted 6 of them as 1, and 9 of them as 0.

It means, for 6 customers, the actual churn value were 1 in test set,
and classifier also correctly predicted those as 1. However, while the
actual label of 9 customers were 1, the classifier predicted those as 0,
which is not very good. We can consider it as error of the model for
first row.

What about the customers with churn value 0? Lets look at the second
row. It looks like there were 25 customers whom their churn value were
0.

The classifier correctly predicted 24 of them as 0, and one of them
wrongly as 1. So, it has done a good job in predicting the customers
with churn value 0. A good thing about confusion matrix is that shows
the model’s ability to correctly predict or separate the classes. In
specific case of binary classifier, such as this example, we can
interpret these numbers as the count of true positives, false positives,
true negatives, and false negatives.

.. code:: ipython3

    print (classification_report(y_test, yhat))


Based on the count of each section, we can calculate precision and
recall of each label:

-  **Precision** is a measure of the accuracy provided that a class
   label has been predicted. It is defined by: precision =
   TP / (TP + FP)

-  **Recall** is true positive rate. It is defined as: Recall =
    TP / (TP + FN)

So, we can calculate precision and recall of each class.

**F1 score:** Now we are in the position to calculate the F1 scores for
each label based on the precision and recall of that label.

The F1 score is the harmonic average of the precision and recall, where
an F1 score reaches its best value at 1 (perfect precision and recall)
and worst at 0. It is a good way to show that a classifer has a good
value for both recall and precision.

And finally, we can tell the average accuracy for this classifier is the
average of the F1-score for both labels, which is 0.72 in our case.

log loss
~~~~~~~~

Now, lets try **log loss** for evaluation. In logistic regression, the
output can be the probability of customer churn is yes (or equals to 1).
This probability is a value between 0 and 1. Log loss( Logarithmic loss)
measures the performance of a classifier where the predicted output is a
probability value between 0 and 1.

.. code:: ipython3

    from sklearn.metrics import log_loss
    log_loss(y_test, yhat_prob)

Practice
--------

Try to build Logistic Regression model again for the same dataset, but
this time, use different **solver** and **regularization** values? What
is new **logLoss** value?

.. code:: ipython3

    # write your code here
    


Double-click **here** for the solution.

.. raw:: html

   <!-- Your answer is below:
       
   LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
   yhat_prob2 = LR2.predict_proba(X_test)
   print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))

   -->

Want to learn more?
-------------------

IBM SPSS Modeler is a comprehensive analytics platform that has many
machine learning algorithms. It has been designed to bring predictive
intelligence to decisions made by individuals, by groups, by systems –
by your enterprise as a whole. A free trial is available through this
course, available here: `SPSS
Modeler <http://cocl.us/ML0101EN-SPSSModeler>`__.

Also, you can use Watson Studio to run these notebooks faster with
bigger datasets. Watson Studio is IBM's leading cloud solution for data
scientists, built by data scientists. With Jupyter notebooks, RStudio,
Apache Spark and popular libraries pre-packaged in the cloud, Watson
Studio enables data scientists to collaborate on their projects without
having to install anything. Join the fast-growing community of Watson
Studio users today with a free account at `Watson
Studio <https://cocl.us/ML0101EN_DSX>`__

Thanks for completing this lesson!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notebook created by: Saeed Aghabozorgi

.. raw:: html

   <hr>

Copyright © 2018 `Cognitive Class <https://cocl.us/DX0108EN_CC>`__. This
notebook and its source code are released under the terms of the `MIT
License <https://bigdatauniversity.com/mit-license/>`__.​
