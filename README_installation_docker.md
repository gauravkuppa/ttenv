## Using this folder on Docker

- docker build -t adfq .
- docker run -it --gpus all --net host --ipc host adfq bash
- conda create -n adfq_venv python tensorflow-gpu=1.14 -y
- conda activate adfq_venv
- pip install pyyaml==5.4.1 scipy numpy tabulate matplotlib gym[atari,classic_control] tqdm joblib zmq dill progressbar2 mpi4py cloudpickle click opencv-python wandb filterpy scikit-image
- git clone https://github.com/gauravkuppa/ttenv.git
- cd ttenv/
- git submodule update --init --recursive
- git checkout headless
- source setup
- cd ADFQ && sourcesetup
 