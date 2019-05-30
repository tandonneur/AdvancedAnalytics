import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
        name="AdvancedAnalytics", 
        version="0.0.5", 
        author="Edward R Jones", 
        author_email="ejones@tamu.edu", 
        url="http://github.com/tandonneur/AdvancedAnalytics", 
        description="Python support for 'The Art and Science of Data Analytics'",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
        ],
)