# Digit-Classifier

An interactive web application to recognize the hand written digits of the user. The pretrained resnet18 CNN model was fine-tuned on the MNIST hand written digits for recognizing the hand written digits of the user.

# Installation

- 1. Install Python 3.12, PostgresSQL and Docker (using admin or root permission) on your linux machine.
- 2. Create a ‘project’ directory on your linux machine.
- 3. Download the following files under /project:
	 - Dockerfile (Docker script for building an image)
	 - Docker-compose.yml (Docker script to setup multiple containers) 
	 - app.py (Streamlit web application interface)
   - requirements.txt (Python libraries required to build the web app)  
	 - resnet18_mnist.pth (the fine-tuned resnet18 model)
   - init.sql (SQL script to setup the backend Postgres database)
	 - digitclassifier.py (Python code to fine-tune the pretrained resnet18 model)

# Running the Web Application

- 1. With root or administer permission, build containers and run the web app:
      ```
        docker-compose up --build -d
      ```
- 2. To access the web interface, open a web browser e.g. Chrome and go to: 
      ```
         <ip_address>:8051
     ```
    where <ip_address> is the IP version 4 address of your machine. 

- 3. Draw a digit between 0 to 9 in the drawing area and input its true label. Then, click the 'predict' button. The predicted digit is displayed together with the confidence of the prediction within [0,1]. The results are logged in the backend database.
  
- 4. To stop the web application:
  ```
  exit         
  ```
  or 
  ```
  docker-compose down        
  ```

If the web app cannot be stopped, find its process id by: 
```
ps aux | grep streamlit 
```
Then, 
```
kill -9 <pid>
```
# License

GNU GLP version 3
