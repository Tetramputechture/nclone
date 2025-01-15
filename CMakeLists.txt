cmake_minimum_required(VERSION 3.20)
project(NCloneCpp LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.0
    GIT_SHALLOW ON)

# Configure SFML options
set(BUILD_SHARED_LIBS TRUE)
set(SFML_BUILD_EXAMPLES FALSE)
set(SFML_BUILD_DOC FALSE)
set(SFML_BUILD_AUDIO TRUE)
set(SFML_BUILD_GRAPHICS TRUE)
set(SFML_BUILD_WINDOW TRUE)
set(SFML_BUILD_NETWORK FALSE)

FetchContent_MakeAvailable(SFML)

add_executable(main 
    src/main.cpp
    src/physics/physics.cpp
    src/physics/segment.cpp
)
target_link_libraries(main PRIVATE sfml-graphics)
target_include_directories(main PRIVATE ${SFML_SOURCE_DIR}/include src)
