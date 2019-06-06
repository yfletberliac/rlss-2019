FROM tensorflow/tensorflow:latest-py3

ENV BUILD_PACKAGES="\
        git \
        wget \
        curl" \
    PIP_PACKAGES="\
#        h5py \
        numpy \
        pandas \
#        scipy \
        seaborn \
        matplotlib \
#        gym \
        gym[atari] \
        git+https://github.com/eleurent/highway-env \
        opencv-python \
        https://download.pytorch.org/whl/cpu/torch-1.1.0-cp35-cp35m-linux_x86_64.whl \
        https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp35-cp35m-linux_x86_64.whl" \
    JUPYTER_CONFIG_DIR=/home/.ipython/profile_default/startup \
    LANG=C.UTF-8

RUN set -ex; \
    apt-get update -y; \
    apt-get upgrade -y; \
    apt-get install -y --no-install-recommends ${BUILD_PACKAGES}; \
    apt-get update -y; \
    apt-get install -y libgtk2.0-dev; \
    pip install -U -V pip; \
    pip install -U -v setuptools wheel; \
    pip install -U -v ${PIP_PACKAGES}; \
    apt-get remove --purge --auto-remove -y ${BUILD_PACKAGES}; \
    apt-get clean; \
    apt-get autoclean; \
    apt-get autoremove; \
    rm -rf /tmp/* /var/tmp/*; \
    rm -rf /var/lib/apt/lists/*; \
    pip install jupyter && jupyter nbextension enable --py widgetsnbextension; \
    mkdir -p ${JUPYTER_CONFIG_DIR}; \
    echo "import warnings" | tee ${JUPYTER_CONFIG_DIR}/config.py; \
    echo "warnings.filterwarnings('ignore')" | tee -a ${JUPYTER_CONFIG_DIR}/config.py; \
    echo "c.NotebookApp.token = u''" | tee -a ${JUPYTER_CONFIG_DIR}/config.py

EXPOSE 8888

CMD [ "jupyter", "notebook", "--port=8888", "--no-browser", \
    "--allow-root", "--ip=0.0.0.0", "--NotebookApp.token=" ]
