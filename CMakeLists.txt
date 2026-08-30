cmake_minimum_required(VERSION 3.16)

project(DinoM2M VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DINO_M2M_BUILD_GMS "Build the many-to-many GMS CSV filter" ON)
option(DINO_M2M_BUILD_LORANSAC "Build the many-to-many relative-pose estimator" ON)

include(CTest)

if(DINO_M2M_BUILD_LORANSAC)
    add_subdirectory(m2m_loransac)
endif()

if(DINO_M2M_BUILD_GMS)
    add_subdirectory(gms_filter)
endif()

if(BUILD_TESTING AND TARGET DinoM2M::hopcroft_karp)
    add_executable(test_hopcroft_karp tests/cpp/test_hopcroft_karp.cpp)
    target_link_libraries(test_hopcroft_karp PRIVATE DinoM2M::hopcroft_karp)
    add_test(NAME hopcroft_karp COMMAND test_hopcroft_karp)
endif()

if(BUILD_TESTING AND TARGET DinoM2M::loransac)
    add_executable(test_relative_pose_m2m tests/cpp/test_relative_pose_m2m.cpp)
    target_link_libraries(test_relative_pose_m2m PRIVATE DinoM2M::loransac)
    add_test(NAME relative_pose_m2m_guards COMMAND test_relative_pose_m2m)
endif()
