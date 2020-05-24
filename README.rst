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

The API and documentation for all classes and examples are available at https://github.com/tandonneur/AdvancedAnalytics/. 

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
        "Salary":         [DT.Interval, (20000.0, 2000000.0)],
        "Department":     [DT.Nominal, ("HR", "Sales", "Marketing")] 
        "Classification": [DT.Nominal, (1, 2, 3, 4, 5)]
        "Years":          [DT.Interval, (18, 60)] }
    # Preprocess data from data frame df
    rie = ReplaceImputeEncode(data_map=data_map, interval_scaling=None,
                              nominal_encoding= "SAS", drop=True)
    encoded_df = rie.fit_transform(df)
    y = encoded_df["Salary"]
    X = encoded_df.drop("Salary", axis=1)
    dt = DecisionTreeRegressor(criterion= "gini", max_depth=4,
                                min_samples_split=5, min_samples_leaf=5)
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

Text
    Classes for Text Analytics
        * text_analysis support for topic analysis
        * text_plot for word clouds
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
    conda install -c dr.jones AdvancedAnalytics

General Dependencies
    There are dependencies.  Most classes import one or more modules from    
    **Sci-Learn**, referenced as *sklearn* in module imports, and 
    **StatsModels**.  These are both installed with the current version
    of **anaconda**.

Installed with AdvancedAnalytics
    Most packages used by **AdvancedAnalytics** are automatically 
    installed with its installation.  These consist of the following 
    packages.

        * statsmodels
        * scikit-learn
        * scikit-image
        * nltk
        * pydotplus

Other Dependencies
    The *Tree* and *Forest* modules plot decision trees and importance
    metrics using **pydotplus** and the **graphviz** packages.  These
    should also be automatically installed with **AdvancedAnalytics**.

    However, the **graphviz** install is sometimes not fully complete 
    with the conda install.  It may require an additional pip install.

    .. code-block:: python

        pip install graphviz

Text Analytics Dependencies
    The *TextAnalytics* module uses the **NLTK**, **Sci-Learn**, and 
    **wordcloud** packages.  Usually these are also automatically 
    installed automatically with **AdvancedAnalytics**.  You can verify 
    they are installed using the following commands.

    .. code-block:: python

        conda list nltk
        conda list sci-learn
        conda list wordcloud

    However, when the **NLTK** package is installed, it does not 
    install the data used by the package.  In order to load the
    **NLTK** data run the following code once before using the 
    *TextAnalytics* module.

    .. code-block:: python

        #The following NLTK commands should be run once
        nltk.download("punkt")
        nltk.download("averaged_preceptron_tagger")
        nltk.download("stopwords")
        nltk.download("wordnet")

    The **wordcloud** package also uses a little know package
    **tinysegmenter** version 0.3.  Run the following code to ensure
    it is installed.

    .. code-block:: python

        conda install -c conda-forge tinysegmenter==0.3
        # or
        pip install tinysegmenter==0.3

Internet Dependencies
    The *Internet* module contains a class *scrape* which has some   
    functions for scraping newsfeeds. Some of these use the 
    **newspaper3k** package.  It should be automatically installed with 
    **AdvancedAnalytics**.

    However, it also uses the package **newsapi-python**, which is not 
    automatically installed.  If you intended to use this news scraping
    scraping tool, it is necessary to install the package using the 
    following code:

    .. code-block:: python

        conda install -c conda-forge newsapi
        # or
        pip install newsapi

    In addition, the newsapi service is sponsored by a commercial company
    www.newsapi.com.  You will need to register with them to obtain an 
    *API* key required to access this service.  This is free of charge 
    for developers, but there is a fee if *newsapi* is used to broadcast 
    news with an application or at a website.

Code of Conduct
---------------

Everyone interacting in the AdvancedAnalytics project's codebases, issue trackers, chat rooms, and mailing lists is expected to follow the PyPA Code of Conduct: https://www.pypa.io/en/latest/code-of-conduct/ .


