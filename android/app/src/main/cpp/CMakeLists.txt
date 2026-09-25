cmake_minimum_required(VERSION 3.22.1)
project(llama_android C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -flto -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -fPIC")

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/CMakeLists.txt")
    add_subdirectory(llama_cpp)
    set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp")
else()
    include(FetchContent)
    FetchContent_Declare(
        llama_cpp
        GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
        GIT_TAG        b3600
    )
    FetchContent_MakeAvailable(llama_cpp)
    set(LLAMA_DIR "${llama_cpp_SOURCE_DIR}")
endif()

add_library(llama-android SHARED
    llama_jni.cpp
)

target_include_directories(llama-android PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_DIR}/src
)

find_library(log-lib log)
find_library(android-lib android)
find_library(dl-lib dl)

target_link_libraries(llama-android
    llama
    ggml
    ${log-lib}
    ${android-lib}
    ${dl-lib}
)
