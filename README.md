# testcode

Building a Docker image (assuming your Dockerfile is current working directory):
> docker build -t inlaplus .

# Install INLAPLUS

## Step 1: install sarus:

> sudo apt update

> sudo apt -y install squashfs-tools

> mkdir /opt/sarus/ 

> cd /opt/sarus/1.5.0-Release

> sudo ./configure_installation.sh

> export PATH=/opt/sarus/1.5.0-Release/bin:${PATH}

It is recommended to add "export PATH=/opt/sarus/1.5.0-Release/bin:${PATH}" to ".bashrc" file, otherwise you need to run it everytime you use INLAPLUS.

## Step 2: pull inlaplus container using:

> sarus pull esmailabdulfattah/inlaplus:130822

## Step 3 using R: install R-INLAPLUS Package:

Run the following two lines:

> library(devtools)
>
> remotes::install_github("esmail-abdulfattah/INLAPLUS")

## Step 3 using Python: 

You need to download the Python folder in this github. It is at R/Python.

Create your own directory and add the pyhton folder to it.

In this folder you have 4 main files: 


## Step 4: run INLAPLUS from R

> mpirun -N 1 -n 2 sarus run --mpi --workdir=/home/abdulfe/R/x86_64-pc-linux-gnu-library/4.2/INLAPLUS/ esmailabdulfattah/inlaplus_test:latest /software/testcode/output

# Run INLAPLUS using Python:
