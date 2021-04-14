# Translation-Station-API
The API for my final year project the Translation station. Built in Python, using Tensorflow and Keras, three different recurrent neural networks have been trained to translate from English to French. There is one endpoint `/translate/<model_id>/<sentance>` used to translate text using the specified model and given english sentence. It returns both the English and French translation in a JSON object. 
## Prerequisites
1. Python 3.7
2. [Pip](https://pip.pypa.io/en/stable/installing/) if it isn't installed when you install Python (used to install Pipenv)

You can check both have been installed by running the following in a terminal:  

`python --version`  
`pip --version`
## Install
1. Clone this repositry via git
2. This project uses [Pipenv](https://pipenv.pypa.io/en/latest/) for package management. To install pipenv run the following command  
`pip install pipenv`  
3. install the projects dependancies  
`Pipenv install`

## Starting Development
To start the API locally run the following command  
`pipenv run flask run`

