cmake_minimum_required(VERSION 3.10)
project(MNIST_Digit_Recognition)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
find_package(TensorFlow REQUIRED)

add_executable(mnist_digit_recognition main.cpp)
target_link_libraries(mnist_digit_recognition ${OpenCV_LIBS} tensorflow_cc tensorflow_framework)
