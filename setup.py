from setuptools import setup, find_packages

setup(
    name='sim_data_interface',
    version='0.0.1',
    description='Issac Dataset Interface',
    packages=find_packages(include=['sim_data_interface', "yolo_to_yolo", "general_classifier"])
)