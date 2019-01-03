# deployObjectDetection
Utility class around C++ TensorFlow for multi-purpose deep learning tasks deployment.

Check my [medium story](https://medium.com/@mohamedtamer92/tensorflow-how-to-export-freeze-models-with-python-api-and-deploy-object-detection-models-with-a6bbb74afe1c) for detailed documentation and other background information

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* OpenCV C++
* CUDA
* cuDNN
* Bazel
* TensorFlow C++
* Protobuf
* Eigen

### Installing
* Download Inception network
```
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -C tensorflow/examples/label_image/data -xz
```
* Do a full-text search for `TODO` in the code and make changes accordingly.
* Run [main.cpp](main.cpp)

## Built With

* [CMake](https://cmake.org/) - Build System
* [GCC](https://gcc.gnu.org/) - Compiler

## Contributing

Please open an issue detailing your changes or feature requests so we can discuss it

## Authors

**Mohamed Abulazm** - [GitHub](https://github.com/Fattouh92), [LinkedIn](https://linkedin.com/in/fattouh92)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
