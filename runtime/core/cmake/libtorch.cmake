if(TORCH)
  add_definitions(-DUSE_TORCH)
  if(NOT ANDROID)
    if(GPU)
    set(CUDA_ARCH_FLAGS "-gencode arch=compute_80,code=sm_80")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")

      if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        message(FATAL_ERROR "GPU is supported only Linux, you can use CPU version")
      else()
        add_definitions(-DUSE_GPU)
      endif()
    endif()

    if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
      if(${CMAKE_BUILD_TYPE} MATCHES "Release")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.13.0%2Bcpu.zip")
        set(URL_HASH "SHA256=bece54d36377990257e9d028c687c5b6759c5cfec0a0153da83cf6f0f71f648f")
      else()
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.13.0%2Bcpu.zip")
        set(URL_HASH "SHA256=3cc7ba3c3865d86f03d78c2f0878fdbed8b764359476397a5c95cf3bba0d665a")
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      if(CXX11_ABI)
        if(NOT GPU)
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip")
          set(URL_HASH "SHA256=d52f63577a07adb0bfd6d77c90f7da21896e94f71eb7dcd55ed7835ccb3b2b59")
        else()
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.2.1%2Bcu118.zip")
          set(URL_HASH "SHA256=80f089939de20e68e3fcad4dfa72a26c8bf91b5e77b11042f671f39ebac35865")
        endif()
      else()
        if(NOT GPU)
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.0%2Bcpu.zip")
          set(URL_HASH "SHA256=bee1b7be308792aa60fc95a4f5274d9658cb7248002d0e333d49eb81ec88430c")
        else()
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.2.1%2Bcu118.zip")
          set(URL_HASH "SHA256=05c8fa6f68e1b45982c9b50e7410ef3ce2d415b6c0d76b7e79162d8740c1a63a")
        endif()
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip")
      set(URL_HASH "SHA256=1ce371a37d3871761d3f0fcf4f4d49267a7b04ebc5152e1be325605f97c80765")
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
      add_definitions(-DIOS)
    else()
      message(FATAL_ERROR "Unsupported System '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux', 'Darwin' or 'iOS')")
    endif()

    # iOS use LibTorch from pod install
    if(NOT IOS)
      FetchContent_Declare(libtorch
        URL      ${LIBTORCH_URL}
        URL_HASH ${URL_HASH}
      )
      FetchContent_MakeAvailable(libtorch)
      find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")
    endif()

    if(MSVC)
      file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
      file(COPY ${TORCH_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
    endif()
  else()
    # Change version in runtime/android/app/build.gradle.
    file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
    file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")
    find_library(PYTORCH_LIBRARY pytorch_jni
      PATHS ${PYTORCH_LINK_DIRS}
      NO_CMAKE_FIND_ROOT_PATH
    )
    find_library(FBJNI_LIBRARY fbjni
      PATHS ${PYTORCH_LINK_DIRS}
      NO_CMAKE_FIND_ROOT_PATH
    )
    include_directories(
      ${PYTORCH_INCLUDE_DIRS}
      ${PYTORCH_INCLUDE_DIRS}/torch/csrc/api/include
    )
  endif()
endif()
