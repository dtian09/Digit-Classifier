# Digit-Classifier

An interactive web application developed using PyTorch (deep learning), Streamlit (Web application framework), PostgresSQL (database) and Docker (containerization) to recognize the hand written digits of the user. The pretrained resnet18 CNN model [1] was fine-tuned on the MNIST hand written digits database [2] for recognizing the hand written digits of the user.

# Installation

- 1. Install Python 3.10 or higher version, PostgresSQL and Docker (using admin or root permission) on your linux machine.
- 2. Create a `project' directory on your linux machine.
- 3. Download the following files under the `project' directory:
	 - Dockerfile (Docker script for building an image)
	 - Docker-compose.yml (Docker script to setup multiple containers) 
	 - app.py (Web Application Code)
   - requirements.txt (Python libraries required to build the web application)  
	 - resnet18_mnist.pth (the fine-tuned resnet18 CNN model)
   - init.sql (SQL script to setup the backend Postgres database)
	 - digitclassifier.py (Python code to fine-tune the pretrained resnet18 model)

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
kill -9 <pid>
```
# References:

- [1] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [2] [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) 
