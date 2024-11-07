FROM giovtorres/slurm-docker-cluster:latest

RUN set -ex \
    && yum makecache \
    && yum -y update \
    && yum -y install iproute \
    && yum clean all \
    && rm -rf /var/cache/yum

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -f -b -p /opt/anaconda && \
    /opt/anaconda/bin/conda clean -tipy && \
    rm -f miniconda.sh
ENV PATH /opt/anaconda/bin:$PATH

COPY environment.yml .
RUN conda env create --file environment.yml

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
