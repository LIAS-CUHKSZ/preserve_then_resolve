cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(DinoM2MGMS VERSION 0.1.0 LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    include(CTest)

endif()

include(GNUInstallDirs)

if(NOT TARGET DinoM2M::hopcroft_karp)
    add_library(m2m_hopcroft_karp STATIC ../m2m_loransac/src/hopcroft_karp.cpp)
    add_library(DinoM2M::hopcroft_karp ALIAS m2m_hopcroft_karp)
    target_compile_features(m2m_hopcroft_karp PUBLIC cxx_std_17)
    target_include_directories(m2m_hopcroft_karp PUBLIC ../m2m_loransac/include)
endif()

find_package(OpenCV REQUIRED COMPONENTS core features2d)

add_library(gms_m2m INTERFACE)
add_library(DinoM2M::gms ALIAS gms_m2m)
target_include_directories(gms_m2m INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(gms_m2m INTERFACE
    DinoM2M::hopcroft_karp
    opencv_core
    opencv_features2d)

add_executable(gms_filter_csv_m2m src/gms_filter_csv_m2m.cpp)
target_link_libraries(gms_filter_csv_m2m PRIVATE DinoM2M::gms)
install(TARGETS gms_filter_csv_m2m RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES LICENSE.GMS DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/DinoM2M")

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_gms_cli.cmake")
    add_test(NAME gms_cli_guards
        COMMAND "${CMAKE_COMMAND}"
            "-DGMS_EXE=$<TARGET_FILE:gms_filter_csv_m2m>"
            "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/gms_cli_test"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_gms_cli.cmake")
endif()
