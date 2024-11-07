FROM giovtorres/slurm-docker-cluster:latest

RUN sed -i 's|mirrorlist=http://mirrorlist.centos.org|#mirrorlist=http://mirrorlist.centos.org|g' /etc/yum.repos.d/CentOS-Base.repo && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Base.repo && \
    yum makecache && \
    yum -y update && \
    yum -y install iproute2 && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -f -b -p /opt/conda && \
    /opt/conda/bin/conda clean -tipy && \
    /opt/conda/bin/conda install python=3.11 && \
    rm -f miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
