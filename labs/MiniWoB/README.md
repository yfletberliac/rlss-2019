Setup process:

Install docker, current user need to have rights to start container (for linux, belong to the `docker` group)

Install go development environment (required by universe)

*Note*: python 3.7 won't work due to universe's usage of `async` keyword
```
docker pull shmuma/miniwob
conda create -n miniwob python=3.6
conda activate miniwob
pip install gym
pip install universe
conda install pytorch torchvision -c pytorch
pip install ptan
pip install opencv_python
```

`pip install universe` - this step will likely require more dependencies to be installed, like libjpeg-turbo, etc
