FROM krccl/openmpi_base:403
RUN apt-get update && \
    apt-get install -y  wget git cmake && \
    apt-get install -y liblapack-dev libblas-dev

WORKDIR /software
RUN wget https://bitbucket.org/blaze-lib/blaze/get/v3.8.1.tar.gz -O blaze-v3.8.1.tar.gz && \
    wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    git clone  https://github.com/esmail-abdulfattah/LBFGSpp.git && \
    git clone -b boost-1.71.0 --recursive https://github.com/boostorg/boost.git
# Install boost
RUN cd boost && \
    ./bootstrap.sh --prefix=/usr/local && \
    echo "using mpi : mpicxx ; \n" >> /software/boost/user-config.jam && \
    ./b2 --user-config=user-config.jam  cxxflags="-std=c++11"
# Install blaze and eigne
WORKDIR /software
RUN tar xvf blaze-v3.8.1.tar.gz && \
    tar xvf eigen-3.4.0.tar.gz
RUN cd blaze-lib-blaze-5074d1f16d4b && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ .. && \
    make install
WORKDIR /software
RUN cd eigen-3.4.0  && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ .. && \
    make install
# Install LBFGSpp
WORKDIR /software
RUN cd LBFGSpp && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ .. && \
    make install

ENV LD_LIBRARY_PATH /software/boost/stage/lib:/usr/local/lib:$LD_LIBRARY_PATH
# download mkl
RUN cd && \
    cd /usr/local/ && \
    wget 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=10RMah6sVf1Arx4UH83bVrOXjlyVQfVIJ' -O MKL && \
    tar xvf MKL 
WORKDIR /software
RUN wget 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=10RMah6sVf1Arx4UH83bVrOXjlyVQfVIJ' -O MKL && \
    tar xvf MKL 
# Build own Inlaplus
WORKDIR /software
RUN rm -rf blaze* eigen* LBFGSpp*
RUN git clone -b master https://github.com/testcodesprojects/testcode.git && \
    cd testcode && \
    make VERBOSE=1 && \
    cd .. && \
    rm -r testcode
ENV PATH /software/testcode:$PATH
