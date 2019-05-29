from distutils.core import setup
desc= """\
AdvancedAnalytics Package
==========================
This package is used in conjunction with the book:
    'The Art and Science of Data Analytics'.
It provides easy to use functions for reporting and producing results for
many standard machine learning models.
"""

setup(name='AdvancedAnalytics', 
      version='1.0', 
      author='Edward R Jones', 
      author_email='ejones@tamu.edu', 
      url='http://bookwebsite.org', 
      long_description=desc, 
      py_modules=['Calculate', 'DecisionTree', 'linreg', 'logreg', 
                  'NeuralNetwork', 'News', 'Sentiment', 'TextAnalytics'],)