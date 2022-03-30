FROM doduo1.umcn.nl/uokbaseimage/diag:tf2.5-pt1.9-v2
FROM doduo1.umcn.nl/uokbaseimage/tensorflow_pytorch_python3:3

COPY --chown=user:user ./code /home/user/

RUN pip install --upgrade pip
RUN pip install --upgrade tensorflow
RUN pip install --upgrade keras
RUN pip install --upgrade h5py
RUN pip install --upgrade numpy


WORKDIR /home/user
User user

ENTRYPOINT ["python3.7", "-u", "main.py"]
