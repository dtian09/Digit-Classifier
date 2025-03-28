# Digit-Classifier

An interactive web application developed using PyTorch (deep learning), Streamlit (Web application framework), PostgresSQL (database) and Docker (containerization) to recognize the hand written digits of the user. The pretrained ResNet-18 CNN model [1] was fine-tuned on the MNIST hand written digits database [2] for recognizing the hand written digits of the user.

 # Installation

- 1. Install Python 3.10 or higher version, PostgresSQL and Docker (using admin or root permission) on your linux machine.
- 2. Create a `project' directory on your linux machine.
- 3. Download the following files under the `project' directory:
     - digitclassifier.py (Python code to fine-tune the pretrained resnet18 model and save it as 'resnet18_mnist.pth')
     - Dockerfile (Docker script for building an image)
     - Docker-compose.yml (Docker script to setup multiple containers) 
     - app.py (Web Application Code)
     - requirements.txt (Python libraries required to build the web application)
     - init.sql (SQL script to setup the backend Postgres database)
           
# Fine-tune the Pretrained ResNet-18 Model

- 1. Install the Python virtual environment module venv:
     ```
     $ apt install python3-venv
     ```
- 2. Go to the 'project' directory:
     ```
     $ cd project
     ```
- 3. Create a virtual environment called my_env:
     ```
     $ python -m venv my_env
     ```
- 4. Active my_env:
     ```
     $ source my_env/bin/activate
     ```
- 5. Install PyTorch, torchvision, scikit-learning and Numpy Python libraries in my_env:
     ```
     $ pip install torch torchvision scikit-learn numpy
     ```
- 6. Fine-tune the pretrained RestNet-18 model:
    ```
    $ python digitclassifier.py
    ``` 
    The fine-tuned model is saved as 'resnet18_mnist.pth'.
- 7. Check that the model is now under the current directory ('project'):
    ```
    $ ls | grep resnet18_mnist
    ```
- 8. Deactivate my_env (i.e. exit my_env):
    ```
    $ deactivate
    ```	
# Running the Web Application

- 1. With root or administer permission, build containers and run the web app:
      ```
       $ docker-compose up --build -d
      ```
- 2. To access the web interface, open a web browser e.g. Chrome and enter the address: 
      ```
       http://<ip_address>:8501
      ```
    where <ip_address> is the IP version 4 address of your linux machine. 

- 3. Draw a digit between 0 to 9 in the drawing area and input its true label. Then, click the 'predict' button. The predicted digit is displayed together with the confidence of the prediction within [0,1]. The results are logged in the backend database.
  
- 4. To stop the web application:
  ```
  $ exit         
  ```
  or 
  ```
  $ docker-compose down        
  ```

If the web app cannot be stopped, find its process id by: 
```
$ ps aux | grep streamlit 
```
Then, 
```
$ kill -9 <pid>
```
# References:

- [1] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [2] [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) 
