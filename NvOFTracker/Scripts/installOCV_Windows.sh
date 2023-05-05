#!/bin/bash -e
myRepo=$(pwd)
if [  ! -d "$myRepo/opencv"  ]; then
    echo "cloning opencv"
    git clone -b 4.5.2 https://github.com/opencv/opencv.git
    mkdir Build
    mkdir Build/opencv
    mkdir Install
    mkdir Install/opencv
else
    cd opencv
    git pull --rebase
    cd ..
fi
if [  ! -d "$myRepo/opencv_contrib"  ]; then
    echo "cloning opencv_contrib"
    git clone -b 4.5.2 https://github.com/opencv/opencv_contrib.git
    mkdir Build/opencv_contrib
else
    cd opencv_contrib
    git pull --rebase
    cd ..
fi
RepoSource=opencv
pushd Build/$RepoSource
CMAKE_OPTIONS='-DOPENCV_CUDA_FORCE_BUILTIN_CMAKE_MODULE:BOOL=1 -DBUILD_opencv_ts:BOOL=0 -DWITH_WIN32UI:BOOL=0 -DBUILD_opencv_world:BOOL=0 -DWITH_PROTOBUF:BOOL=0 -DCPU_BASELINE:STRING=SSE3 -DBUILD_opencv_aruco:BOOL=1 -DBUILD_opencv_python_bindings_generator:BOOL=0 -DWITH_CUDA:BOOL=1 -DBUILD_WEBP:BOOL=0 -DWITH_IMGCODEC_HDR:BOOL=0 -DBUILD_PACKAGE:BOOL=0 -DWITH_OPENCLAMDFFT:BOOL=0 -DWITH_OPENEXR:BOOL=0 -DBUILD_opencv_features2d:BOOL=1 -DBUILD_ZLIB:BOOL=0 -DWITH_OPENCLAMDBLAS:BOOL=0 -DBUILD_opencv_flann:BOOL=1 -DBUILD_TIFF:BOOL=0 -DWITH_IMGCODEC_PXM:BOOL=0 -DCPU_DISPATCH:STRING="" -DBUILD_JPEG:BOOL=0 -DBUILD_opencv_photo:BOOL=0 -DWITH_LAPACK:BOOL=0 -DBUILD_opencv_python_tests:BOOL=0 -DWITH_OPENCL_D3D11_NV:BOOL=0 -DWITH_PNG:BOOL=0 -DBUILD_opencv_stitching:BOOL=0 -DCUDA_ARCH_BIN:STRING=7.5 -DCUDA_ARCH_PTX:STRING=7.5 -DWITH_MSMF:BOOL=0 -DWITH_OPENCL:BOOL=0 -DBUILD_opencv_ml:BOOL=0 -DWITH_MSMF_DXVA:BOOL=0 -DWITH_IMGCODEC_SUNRASTER:BOOL=0 -DWITH_IMGCODEC_PFM:BOOL=0 -DBUILD_PERF_TESTS:BOOL=0 -DWITH_VTK:BOOL=0 -DBUILD_ITT:BOOL=0 -DBUILD_opencv_python3:BOOL=0 -DWITH_QUIRC:BOOL=0 -DWITH_GSTREAMER:BOOL=0 -DBUILD_PNG:BOOL=0 -DBUILD_JAVA:BOOL=0 -DBUILD_opencv_apps:BOOL=0 -DBUILD_JASPER:BOOL=0 -DBUILD_PROTOBUF:BOOL=0 -DWITH_TIFF:BOOL=0 -DBUILD_opencv_cudaoptflow:BOOL=1 -DBUILD_opencv_calib3d:BOOL=1 -DBUILD_opencv_java_bindings_generator:BOOL=0 -DWITH_IPP:BOOL=0 -DWITH_JPEG:BOOL=0 -DWITH_ITT:BOOL=0 -DWITH_WEBP:BOOL=0 -DOPENCV_DNN_CUDA:BOOL=0 -DBUILD_IPP_IW:BOOL=0 -DWITH_NVCUVID:BOOL=0 -DWITH_FFMPEG:BOOL=1 -DBUILD_OPENEXR:BOOL=0 -DWITH_EIGEN:BOOL=0 -DBUILD_opencv_gapi:BOOL=0 -DWITH_JASPER:BOOL=0 -DBUILD_TESTS:BOOL=0'
cmake $CMAKE_OPTIONS -DOPENCV_EXTRA_MODULES_PATH="$myRepo"/opencv_contrib/modules -DCMAKE_INSTALL_PREFIX="$myRepo"/Install/"$RepoSource" "$myRepo/$RepoSource"
echo "************************* $Source_DIR -->release"
cmake --build .  --config release
cmake --build .  --target install --config release
popd