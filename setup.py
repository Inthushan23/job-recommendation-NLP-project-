from setuptools import setup, find_packages

setup(
    name="my_project",
    version="0.1",
    description="Project NLP with recommender and data services",
    packages=find_packages(where="services/data_service/src") + 
             find_packages(where="services/recommender_service/src")
)
