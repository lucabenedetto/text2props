from setuptools import find_packages, setup

setup(
    name='text2props',
    packages=find_packages(),
    version='0.3.0',
    description='Framework for the estimation of questions latent traits from text.',
    author='Luca Benedetto',
    license='gpl-3.0',
    python_requires='>=3.12.3',
    install_requires=[
        'gensim==4.4.0',
        'nltk==3.9.4',
        'numpy==2.4.2',
        'pandas==3.0.0',
        'pyirt==0.3.4',
        'scikit-learn==1.8.0',
        'scipy==1.17.0',
        'textstat==0.7.12',
    ],
)
