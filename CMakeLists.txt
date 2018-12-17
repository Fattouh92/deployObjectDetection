cmake_minimum_required(VERSION 3.12)

project(deployObjectDetection)

set(CMAKE_CXX_STANDARD 14)

# TODO: Provide your own paths to the following libs

#------------------------------------------------------------------
#--------lib opencv------------------------------------------
#------------------------------------------------------------------

find_package(OpenCV REQUIRED PATHS "/home/mohamed/opt_audi/opencv/3.4.1")

if(NOT OpenCV_LIBS)
    message(FATAL_ERROR "OpenCV lib not found. Please specify the OPENCV_DIR")
else(NOT OpenCV_LIBS)
    set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
    message(STATUS "OpenCV lib found. OpenCV Version is ${OpenCV_VERSION}")
    set(AADC_OPENCV_FOUND TRUE)
    message(STATUS "OpenCV Include dir is ${OpenCV_INCLUDE_DIRS}")
endif(NOT OpenCV_LIBS)

#------------------------------------------------------------------
#--------lib protobuf------------------------------------------
#------------------------------------------------------------------

if(UNIX)
    set(protobuf_MODULE_COMPATIBLE TRUE)
    FIND_PACKAGE(Protobuf 3.5.0 REQUIRED)

    if (Protobuf_FOUND)
        message(STATUS "Found Protobuf lib with version ${Protobuf_VERSION}, libs are: ${Protobuf_LIBRARIES}; Include Dir is ${Protobuf_INCLUDE_DIRS}")
    else (Protobuf_FOUND)
        message(FATAL_ERROR "Protobuf lib not found")
    endif (Protobuf_FOUND)
endif(UNIX)

#------------------------------------------------------------------
#--------lib eigen------------------------------------------
#------------------------------------------------------------------

if(UNIX)
    FIND_PACKAGE(Eigen3 3.3.4 REQUIRED PATHS "/home/mohamed/opt_audi/eigen/3.3.4")

    if (Eigen3_FOUND)
        message(STATUS "Found Eigen ${EIGEN3_VERSION_STRING} ; Include Dir is ${EIGEN3_INCLUDE_DIRS}")
    else (Eigen3_FOUND)
        message(FATAL_ERROR "Eigen not found")
    endif (Eigen3_FOUND)
endif(UNIX)

#------------------------------------------------------------------
#--------lib tensorflow------------------------------------------
#------------------------------------------------------------------

if(UNIX)
    set(TENSORFLOW_DIR "/home/mohamed/opt_audi/tensorflow/1.8.0")
    set(TENSORFLOW_INCLUDE_DIRS "${TENSORFLOW_DIR}/include")


    FIND_LIBRARY(TENSORFLOW_LIB NAMES
            tensorflow_cc
            PATHS
            "${TENSORFLOW_DIR}/lib"
            )

    FIND_LIBRARY(TENSORFLOW_FRAMEWORK NAMES
            tensorflow_framework
            PATHS
            "${TENSORFLOW_DIR}/lib"
            )

    set(TENSORFLOW_LIBS "${TENSORFLOW_LIB};${TENSORFLOW_FRAMEWORK}")


    if (TENSORFLOW_LIBS)
        message(STATUS "Found Tensorflow. Tensorflow libs are: ${TENSORFLOW_LIBS}; Include Dir is ${TENSORFLOW_INCLUDE_DIRS}")
    else (TENSORFLOW_LIBS)
        message(FATAL_ERROR "Tensorflow lib not found under ${TENSORFLOW_DIR}")
    endif (TENSORFLOW_LIBS)

endif(UNIX)

#------------------------------------------------------------------
#--------end libs------------------------------------------
#------------------------------------------------------------------

include_directories(${Protobuf_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${TENSORFLOW_INCLUDE_DIRS})

add_executable(deployObjectDetection
        ObjectDetection.h
        ObjectDetection.cpp
        main.cpp)

target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${OpenCV_LIBS} ${Protobuf_LIBRARIES} -Wl,--allow-multiple-definition -Wl,--whole-archive ${TENSORFLOW_LIBS} -Wl,--allow-multiple-definition -Wl,--whole-archive)