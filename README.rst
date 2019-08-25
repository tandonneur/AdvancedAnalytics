AdvancedAnalytics
===================

A collection of python modules, classes and methods for simplifying the use of machine learning solutions.  **AdvancedAnalytics** provides easy access to advanced tools in **Sci-Learn**, **NLTK** and other machine learning packages.  **AdvancedAnalytics** was developed to simplify learning python from the book *The Art and Science of Data Analytics*.

Description
===========

From a high level view, building machine learning applications typically proceeds through three stages:

    1. Data Preprocessing
    2. Modeling or Analytics
    3. Postprocessing

The classes and methods in **AdvancedAnalytics** primarily support the first and last stages of machine learning applications. 

Data scientists report they spend 80% of their total effort in first and last stages. The first stage, *data preprocessing*, is concerned with preparing the data for analysis.  This includes:

    1. identifying and correcting outliers, 
    2. imputing missing values, and 
    3. encoding data. 

The last stage, *solution postprocessing*, involves developing graphic summaries of the solution, and metrics for evaluating the quality of the solution.

Documentation and Examples
============================

The API and documentation for all classes and examples are available at https://github.com/tandonneur/AdvancedAnalytics . 

Usage
=====

Currently the most popular usage is for supporting solutions developed using these advanced machine learning packages:

    * Sci-Learn
    * StatsModels
    * NLTK

The intention is to expand this list to other packages.  This is a simple example for linear regression that uses the data map structure to preprocess data:

.. code-block:: python

    from AdvancedAnalytics.ReplaceImputeEncode import DT
    from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode
    from AdvancedAnalytics.Tree import tree_regressor
    from sklearn.tree import DecisionTreeRegressor, export_graphviz 
    # Data Map Using DT, Data Types
    data_map = {
        “Salary”:         [DT.Interval, (20000.0, 2000000.0)],
        “Department”:     [DT.Nominal, (“HR”, “Sales”, “Marketing”)] 
        “Classification”: [DT.Nominal, (1, 2, 3, 4, 5)]
        “Years”:          [DT.Interval, (18, 60)] }
    # Preprocess data from data frame df
    rie = ReplaceImputeEncode(data_map=data_map, interval_scaling=None,
                              nominal_encoding= “SAS”, drop=True)
    encoded_df = rie.fit_transform(df)
    y = encoded_df[“Salary”]
    X = encoded_df.drop(“Salary”, axis=1)
    dt = DecisionTreeRegressor(criterion= “gini”, max_depth=4
                                min_samples_split=5, min_samples_leaf5)
    dt = dt.fit(X,y)
    tree_regressor.display_importance(dt, encoded_df.columns)
    tree_regressor.display_metrics(dt, X, y)

Current Modules and Classes
=============================

ReplaceImputeEncode
    Classes for Data Preprocessing
        * DT defines new data types used in the data dictionary
        * ReplaceImputeEncode a class for data preprocessing

Regression
    Classes for Linear and Logistic Regression
        * linreg support for linear regressino
        * logreg support for logistic regression
        * stepwise a variable selection class

Tree
    Classes for Decision Tree Solutions
        * tree_regressor support for regressor decision trees
        * tree_classifier support for classification decision trees

Forest
    Classes for Random Forests
        * forest_regressor support for regressor random forests
        * forest_classifier support for classification random forests

NeuralNetwork
    Classes for Neural Networks
        * nn_regressor support for regressor neural networks
        * nn_classifier support for classification neural networks

TextAnalytics
    Classes for Text Analytics
        * text_analysis support for topic analysis
        * sentiment_analysis support for sentiment analysis

Internet
    Classes for Internet Applications
        * scrape support for web scrapping
        * metrics a class for solution metrics

Installation and Dependencies
=============================

**AdvancedAnalytics** is designed to work on any operating system running python 3.  It can be installed using **pip** or **conda**.

.. code-block:: python

    pip install AdvancedAnalytics
    # or
    conda install -c conda-forge AdvancedAnalytics

General Dependencies
    There are dependencies.  Most classes import one or more modules from    
    **Sci-Learn**, referenced as *sklearn* in module imports, and 
    **StatsModels**.  These are both installed in with current versions
    of **anaconda**, a popular application for coding python solutions.

Decision Tree and Random Forest Dependencies
    The *Tree* and *Forest* modules plot decision trees and importance
    metrics using **pydotplus** and the **graphviz** packages.  If these
    are not installed and you are planning to use the *Tree* or *Forest*
    modules, they can be installed using the following code.

    .. code-block:: python

        conda install -c conda-forge pydotplus
        conda install -c conda-forge graphviz
        pip install graphviz

    One note, the second conda install does not complete the install of 
    the graphviz package.  To complete the graphviz install, it is 
    necessary to run the pip install after the conda graphviz install.

Text Analytics Dependencies
    The *TextAnalytics* module is based on the **NLTK** and **Sci-Learn**
    text analytics packages.  They are both installed with the current 
    version of anaconda. 

    However, *TextAnalytics* includes options to produce word clouds, 
    which are graphic displays of the word collections associated with 
    topic or data clusters.  The **wordcloud** package is used to produce
    these graphs.  If you are using the *TextAnalytics* module you can
    install the **wordcloud** package with the following code.

    .. code-block:: python

        conda install -c conda-forge wordcloud

    In addition, data used by the **NLTK** package is not automatically 
    installed with this package.  These data include the text 
    dictionary and other data tables.

    The following nltk.download commands should be run before using 
    **TextAnalytics**. However, it is only necessary to run these once to 
    download and install the data NLTK uses for text analytics.

    .. code-block:: python

        #The following NLTK commands should be run once to 
        #download and install NLTK data.
        nltk.download(“punkt”)
        nltk.download(“averaged_preceptron_tagger”)
        nltk.download(“stopwords”)
        nltk.download(“wordnet”)

Internet Dependencies
    The *Internet* module is contains a class *scrape* which has some   
    functions for scraping newsfeeds. Some of these is based on the 
    **newspaper3k** package.  It can be installed using:

    .. code-block:: python

        conda install -c conda-forge newspaper3k
        # or
        pip install newpaper3k

Code of Conduct
---------------

Everyone interacting in the AdvancedAnalytics project's codebases, issue trackers, chat rooms, and mailing lists is expected to follow the PyPA Code of Conduct: https://www.pypa.io/en/latest/code-of-conduct/ .


