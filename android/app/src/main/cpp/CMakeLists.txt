cmake_minimum_required(VERSION 3.22.1)
project(llama_android C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Оптимизации компилятора под 64-битный ARM (Helio G99 Cortex-A76/A55)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto")

# Автоматическое подключение исходников llama.cpp через FetchContent
include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b3600
    GIT_SHALLOW    TRUE
)

# Отключаем ненужные утилиты для быстрой сборки Android библиотеки
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

add_library(llama-android SHARED
    llama_jni.cpp
)

target_include_directories(llama-android PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/src
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(llama-android
    llama
    ggml
    ${log-lib}
    ${android-lib}
)
