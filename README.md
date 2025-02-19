# BScThesis

## 🏋️‍♂️ Squat Form Analysis with OpenPose & Machine Learning

This project uses **OpenPose**, **Machine Learning**, and **Computer Vision** to analyze squat form for injury prevention. The environment is fully containerized using **Docker** to ensure reproducibility so that we can easily work together.

## 🚀 Getting Started
Follow these steps to set up your local development environment.

### **1. Clone the Repository**
```sh
git clone https://github.com/perujuice/BScThesis.git
cd BScThesis
```

### **2. Install Docker & Docker Compose**
I asume windows here but can't be too hard to figure out alternative methods for another OS

*Download and install <a href="https://www.docker.com/products/docker-desktop/"> Docker Desktop </a>*

#### To verify the installation:
```sh
docker --version
docker-compose --version
```

### **3. 🔨 Build and Run the environment**

#### Build the Docker Image (First Time Only)
```sh
docker-compose build
```

#### ▶ Start the Container
```sh
docker-compose up -d
```

#### 🛠 Access the Running Container
```sh
docker exec -it openpose-ml bash
```

### **4. 🔄 Updating the Environment**
Whenever changes are made to the Docker setup, rebuild the environment:
```sh
docker-compose up --build -d
```
To stop the container:
```sh
docker-compose down
```

## .gitignore note
*The gitignore file is just some example for now but could be more detailed or configured to be more useful as we go!*
