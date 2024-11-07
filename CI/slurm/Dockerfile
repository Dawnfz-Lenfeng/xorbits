FROM giovtorres/slurm-docker-cluster:latest

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -tipsy
ENV PATH=$CONDA_DIR/bin:$PATH
COPY environment.yml .
RUN conda env create --file environment.yml
SHELL ["conda", "run", "-n", "xorbits-jobqueue", "/bin/bash", "-c"]

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
