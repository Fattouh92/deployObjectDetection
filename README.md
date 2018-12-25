# deployObjectDetection
Utility class around C++ TensorFlow for multi-purpose deep learning tasks deployment

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

#### OpenCV
```
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```
#### CUDA
#### cuDNN
#### Bazel
#### TensorFlow

#### Protobuf

Prepare the environment with

    sudo apt-get update
    sudo apt-get install autoconf automake libtool curl make g++ unzip

Next, download and extract the package

    cd /opt
    sudo wget https://github.com/protocolbuffers/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz -O /tmp/protobuf.tar.gz
    sudo mkdir protobuf
    sudo tar -xzf /tmp/protobuf.tar.gz -C protobuf
    sudo chown -R $(whoami):$(whoami) protobuf

Next, configure, make, install and update the shared library path

    cd /opt/protobuf/protobuf-3.5.1
    chmod +x configure
    ./configure
    make -j5
    make check -j5
    sudo make install
    sudo ldconfig

The libraries should have been installed to `/usr/local/lib`, you can check this with

    ls /usr/local/lib/ | grep proto

#### Eigen
http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [CMake](https://cmake.org/) - Build System
* [GCC](https://gcc.gnu.org/) - Compiler

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
