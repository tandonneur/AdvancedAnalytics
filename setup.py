from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
        name="AdvancedAnalytics", 
        version="1.7", 
        author="Edward R Jones", 
        author_email="ejones@tamu.edu", 
        url="https://github.com/tandonneur/AdvancedAnalytics", 
        description="Python support for 'The Art and Science of Data Analytics'",
	keywords="Statistics Art Science Data Analytics",
        long_description=long_description,
        long_description_content_type="text/markdown",
	py_modules=["AdvancedAnalytics" ],
        packages=find_packages(),
	python_requires=">=3.5",
        classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
        ],
)