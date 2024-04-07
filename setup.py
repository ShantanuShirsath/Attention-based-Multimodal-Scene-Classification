from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    hyphen_e_dot = '-e .'
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements


setup(
name='Multimodal_classification',
version= '0.0.1',
author= 'Shantanu',
author_email= 'shirsath99@gmail.com',
packages= find_packages(),
install_requires = get_requirements('requirements.txt')
)