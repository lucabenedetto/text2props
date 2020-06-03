from setuptools import find_packages, setup

setup(
    name='text2props',
    packages=find_packages(),
    version='0.1.0',
    description='Framework for the estimation of questions latent traits from text.',
    author='Luca Benedetto',
    license='',
    python_requires='>=3.7',
    install_requires=[
        'gensim==3.8.1',
        'nltk==3.4.1',
        'numpy==1.18.1',
        'pandas==0.24.2',
        'pyirt==0.3.3.1',
        'scikit-learn==0.21.2',
        'scipy==1.4.1',
        'textstat==0.5.6',
    ],
)
