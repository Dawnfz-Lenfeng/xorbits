FROM giovtorres/slurm-docker-cluster:latest

RUN sed -i 's|mirrorlist=http://mirrorlist.centos.org/?release=7&arch=x86_64&repo=os&infra=container|#mirrorlist=http://mirrorlist.centos.org/?release=7&arch=x86_64&repo=os&infra=container|g' /etc/yum.repos.d/CentOS-Base.repo && \
    sed -i 's|#baseurl=http://mirror.centos.org/centos/7/os/x86_64|baseurl=http://vault.centos.org/centos/7/os/x86_64|g' /etc/yum.repos.d/CentOS-Base.repo && \
    sed -i 's|mirrorlist=https://mirrorlist.centos.org/?release=7&arch=x86_64&repo=updates|#mirrorlist=https://mirrorlist.centos.org/?release=7&arch=x86_64&repo=updates|g' /etc/yum.repos.d/CentOS-Base.repo && \
    sed -i 's|#baseurl=http://mirror.centos.org/centos/7/updates/x86_64|baseurl=http://vault.centos.org/centos/7/updates/x86_64|g' /etc/yum.repos.d/CentOS-Base.repo && \
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
