# Download and Setup

This tutorial will guide installation of a *dockerization* of TensorFlow, PyTorch and other useful libraries on Linux, MacOS and Windows.

# Docker

In this tutorial we will use Docker containers to handle dependencies and run our code.
Docker is a container which allows us to run our code in an encapsulated container.
The language of choice will be python3.

## 1. Installation of Docker CE (all operating systems)

Instructions for installing Docker CE can be found [here](https://docs.docker.com/engine/installation/#installation), the instructions contains guides for most operating systems.

NOTE: on Windows, *Docker Desktop for Windows* is only available for Windows 10 Pro, Enterprise or Education. For the Home edition or previous versions, use [Docker Toolbox](https://docs.docker.com/toolbox/overview/) instead.

## 2. Using dockerhub

After installing docker you are ready to go! The docker image that you will use for this tutorial is an extension of TensorFlow docker image (with PyTorch, TensorFlow, Jupyter Notebook, numpy, pandas, wget, etc.).

To run the [docker image](https://hub.docker.com/r/yfletberliac/rlss2019-docker), type

```bash
docker run -it yfletberliac/rlss2019-docker
```

this starts up a docker container from the `yfletberliac/rlss2019-docker` image.
Where `-it` is required for an interactive session with the docker bash environment.
To exit the interactive environment of the docker container, type `exit` or `ctrl+D`.

## 3. Forwarding port

As the docker system runs independent of your host system, we need to enable port forwarding (for jupyter notebook):
```bash
docker run -it -p 8888:8888 -v path/to/your/rlss2019-hands-on:/mnt/rlss2019-hands-on -d yfletberliac/rlss2019-docker
```

where `-it` is required for an interactive experience with the docker bash environment, `-p` is for port forwarding.

Now, open a new tab in your browser and type localhost:8888 in the browser address bar. Note that you cannot have any other notebooks running simultaneously.

NOTE: on Windows, the port will probably not bind to localhost, instead you must find the port it binds to by typing the following in your docker prompt

```bash
docker-machine ip
```

This should give you an ip that you can replace with localhost.

### 4. Mounting the RLSS2019 workspace directory

Likewise, we need to give the docker instance access to our workspace directory.
First, download this repository with the command

```bash
git clone https://github.com/yfletberliac/rlss2019-hands-on.git
```
We can use the `-v` option for mounting the folder to the docker container.
```bash
docker run -it -p 8888:8888 -v path/to/your/rlss2019-hands-on:/mnt/rlss2019-hands-on -d yfletberliac/rlss2019-docker
```
From within the notebook, click on `/mnt`, click on `rlss2019-hands-on`. You should find the materials for the practical sessions !

NOTE: on Windows, Docker expects that your data volumes will be within `C:\Users`. This is because Docker has limited access to the filesystem on the host computer. This can be changed:
* with Docker for Windows, go to Settings > Share drive
* with Docker Toolbox, follow [this](https://stackoverflow.com/questions/33126271/how-to-use-volume-option-with-docker-toolbox-on-windows?answertab=votes#tab-top)
