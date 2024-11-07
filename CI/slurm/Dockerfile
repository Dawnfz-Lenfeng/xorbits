FROM giovtorres/slurm-docker-cluster:latest

# mirrorlist.centos.org doesn't exists anymore
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/CentOS-*.repo && \
    sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/CentOS-*.repo && \
    sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/CentOS-*.repo && \
    yum makecache && \
    yum -y update && \
    yum -y install iproute && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -f -b -p /opt/conda && \
    /opt/conda/bin/conda clean -tipy && \
    /opt/conda/bin/conda install python=3.11 && \
    rm -f miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
