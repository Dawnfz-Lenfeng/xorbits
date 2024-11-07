FROM giovtorres/slurm-docker-cluster:latest

RUN apt-get update && apt-get install -y iproute2

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -f -b -p /opt/conda && \
    /opt/conda/bin/conda clean -tipy && \
    /opt/conda/bin/conda install python=3.11 && \
    rm -f miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
