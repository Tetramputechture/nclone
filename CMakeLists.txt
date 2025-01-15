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

# Add all source files
add_executable(main 
    src/main.cpp
    src/renderer.cpp
    src/simulation.cpp
    src/ninja.cpp
    src/sim_config.cpp
    
    # Physics files
    src/physics/physics.cpp
    src/physics/segment.cpp
    
    # Entity files
    src/entities/entity.cpp
    src/entities/one_way_platform.cpp
    src/entities/laser.cpp
    src/entities/thwump.cpp
    src/entities/door_base.cpp
    src/entities/door_regular.cpp
    src/entities/door_locked.cpp
    src/entities/door_trap.cpp
    src/entities/drone_base.cpp
    src/entities/drone_chaser.cpp
    src/entities/drone_zap.cpp
    src/entities/launch_pad.cpp
    src/entities/exit_switch.cpp
    src/entities/exit.cpp
    src/entities/gold.cpp
    src/entities/toggle_mine.cpp
    src/entities/bounce_block.cpp
    src/entities/grid_segment_circular.cpp
    src/entities/grid_segment_linear.cpp
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_link_libraries(main PRIVATE sfml-graphics)
target_include_directories(main PRIVATE ${SFML_SOURCE_DIR}/include src)
