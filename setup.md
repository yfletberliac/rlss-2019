# Download and Setup

This tutorial will guide installation of a *dockerization* of TensorFlow, PyTorch and other useful libraries on Linux, MacOS and Windows.

# Docker

In this tutorial we will use [docker](https://www.docker.com/) containers to handle dependencies and run our code.
Docker is a container which allows us to run our code in an encapsulated container.
The language of choice will be python3.

## 1. Installation of docker (all operating systems)

Instructions for installing docker can be found [here](https://docs.docker.com/engine/installation/#installation), the instructions contains guides for most operating systems.

## 2. Using dockerhub

After installing docker you are ready to go! The docker image that you will use for this tutorial is an extension of TensorFlow docker image (with PyTorch, TensorFlow, Jupyter Notebook, numpy, pandas, wget, etc.).

To run the docker, type

```bash
docker run -it yfletberliac/rlss2019-docker
```

this starts up a docker container from the `yfletberliac/rlss2019-docker` image.
Where `-it` is required for an interactive session with the docker bash environment.
To exit the interactive environment of the docker container, type `exit` or `ctrl+D`.

## 3. Forwarding port

As the docker system runs independent of your host system, we need to enable port forwarding (for jupyter notebook) and sharing of directories.

First, download this repository with the command

```bash
git clone https://github.com/yfletberliac/rlss2019-hands-on.git
```

Then, run
```bash
docker run -it -p 8888:8888 -v path/to/your/rlss2019-hands-on:/mnt/rlss2019-hands-on -d yfletberliac/rlss2019-docker
```

where `-it` is required for an interactive experience with the docker bash environment, `-p` is for port forwarding	and `-v` is for mounting your given folder to the docker container.

Now, open a new tab in your browser and type localhost:8888 in the browser address bar. Note that you cannot have any other notebooks running simultaneously.

NOTE: on Windows, the port will probably not bind to localhost, instead you must find the port it binds to by typing the following in your docker prompt

```bash
docker-machine ip
```

This should give you an ip that you can replace with localhost.

From within the notebook, click on `/mnt`, click on `rlss2019-hands-on`. You should find the materials for the practical sessions !
