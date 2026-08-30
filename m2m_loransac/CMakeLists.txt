cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(DinoM2MLORansac VERSION 0.1.0 LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    include(CTest)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(M2M_LORANSAC_BUILD_APPS "Build the dataset runner and toy example" ON)
option(M2M_LORANSAC_USE_BUNDLED_POSELIB "Use ../third_party/PoseLib when available" ON)
option(M2M_LORANSAC_VERBOSE "Enable per-pair RANSAC diagnostic output" OFF)
set(M2M_LORANSAC_POSELIB_SOURCE_DIR "" CACHE PATH "Optional PoseLib source tree")

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

if(NOT TARGET PoseLib::PoseLib)
    if(M2M_LORANSAC_POSELIB_SOURCE_DIR)
        # Keep PoseLib in the install tree as well as the build tree: the
        # exported DinoM2M target has a public PoseLib dependency.
        add_subdirectory("${M2M_LORANSAC_POSELIB_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/PoseLib")
    elseif(M2M_LORANSAC_USE_BUNDLED_POSELIB AND
           EXISTS "${CMAKE_CURRENT_LIST_DIR}/../third_party/PoseLib/CMakeLists.txt")
        set(BUILD_TESTS OFF CACHE BOOL "Disable bundled PoseLib tests" FORCE)
        set(WITH_BENCHMARK OFF CACHE BOOL "Disable bundled PoseLib benchmarks" FORCE)
        set(PYTHON_PACKAGE OFF CACHE BOOL "Disable bundled PoseLib Python bindings" FORCE)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../third_party/PoseLib"
                         "${CMAKE_CURRENT_BINARY_DIR}/PoseLib")
    else()
        find_package(PoseLib CONFIG REQUIRED)
    endif()
endif()

add_library(m2m_hopcroft_karp STATIC src/hopcroft_karp.cpp)
add_library(DinoM2M::hopcroft_karp ALIAS m2m_hopcroft_karp)
set_target_properties(m2m_hopcroft_karp PROPERTIES EXPORT_NAME hopcroft_karp)
target_compile_features(m2m_hopcroft_karp PUBLIC cxx_std_17)
target_include_directories(m2m_hopcroft_karp PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

add_library(m2m_loransac STATIC src/relative_pose_m2m.cpp)
add_library(DinoM2M::loransac ALIAS m2m_loransac)
set_target_properties(m2m_loransac PROPERTIES EXPORT_NAME loransac)
target_compile_features(m2m_loransac PUBLIC cxx_std_17)
target_compile_definitions(m2m_loransac PUBLIC
    "M2M_VERBOSE=$<IF:$<BOOL:${M2M_LORANSAC_VERBOSE}>,1,0>")
target_include_directories(m2m_loransac PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(m2m_loransac PUBLIC DinoM2M::hopcroft_karp Eigen3::Eigen PoseLib::PoseLib)

if(M2M_LORANSAC_BUILD_APPS)
    add_library(m2m_loransac_harness STATIC
        src/config.cpp
        src/geometry.cpp
        src/io.cpp
        src/runner.cpp)
    target_include_directories(m2m_loransac_harness PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
    target_link_libraries(m2m_loransac_harness PUBLIC DinoM2M::loransac)

    add_executable(m2m_loransac_runner apps/run_dataset.cpp)
    target_link_libraries(m2m_loransac_runner PRIVATE m2m_loransac_harness)

    add_executable(m2m_loransac_toy examples/toy_m2m.cpp)
    target_link_libraries(m2m_loransac_toy PRIVATE DinoM2M::loransac)

    if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_loransac_runner.cmake")
        add_test(NAME loransac_runner_guards
            COMMAND "${CMAKE_COMMAND}"
                "-DRUNNER_EXE=$<TARGET_FILE:m2m_loransac_runner>"
                "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/loransac_runner_test"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_loransac_runner.cmake")
    endif()

    if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_match_csv_io.cpp")
        add_executable(test_match_csv_io ../tests/cpp/test_match_csv_io.cpp)
        target_link_libraries(test_match_csv_io PRIVATE m2m_loransac_harness)
        add_test(NAME match_csv_io_fail_closed
            COMMAND test_match_csv_io
                "${CMAKE_CURRENT_BINARY_DIR}/match_csv_io_test")
    endif()

    if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_loransac_csv_fail_closed.cmake")
        add_test(NAME loransac_csv_fail_closed
            COMMAND "${CMAKE_COMMAND}"
                "-DRUNNER_EXE=$<TARGET_FILE:m2m_loransac_runner>"
                "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/loransac_csv_fail_closed_test"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_loransac_csv_fail_closed.cmake")
    endif()

    if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_install_package.cmake")
        add_test(NAME loransac_installed_package
            COMMAND "${CMAKE_COMMAND}"
                "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
                "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/loransac_install_test"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_install_package.cmake")
    endif()
endif()

install(TARGETS m2m_hopcroft_karp m2m_loransac
    EXPORT DinoM2MLORansacTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES
    include/m2m_loransac/hopcroft_karp.h
    include/m2m_loransac/ransac_m2m_impl.h
    include/m2m_loransac/ransac_mode.h
    include/m2m_loransac/relative_pose_m2m.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/m2m_loransac")

if(M2M_LORANSAC_BUILD_APPS)
    install(TARGETS m2m_loransac_runner m2m_loransac_toy
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DinoM2MLORansacConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DinoM2MLORansacConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DinoM2MLORansacConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/DinoM2MLORansac")
install(EXPORT DinoM2MLORansacTargets
    FILE DinoM2MLORansacTargets.cmake
    NAMESPACE DinoM2M::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/DinoM2MLORansac")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DinoM2MLORansacConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DinoM2MLORansacConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/DinoM2MLORansac")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND BUILD_TESTING AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_hopcroft_karp.cpp")
    add_executable(test_hopcroft_karp ../tests/cpp/test_hopcroft_karp.cpp)
    target_link_libraries(test_hopcroft_karp PRIVATE DinoM2M::hopcroft_karp)
    add_test(NAME hopcroft_karp COMMAND test_hopcroft_karp)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND BUILD_TESTING AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_relative_pose_m2m.cpp")
    add_executable(test_relative_pose_m2m ../tests/cpp/test_relative_pose_m2m.cpp)
    target_link_libraries(test_relative_pose_m2m PRIVATE DinoM2M::loransac)
    add_test(NAME relative_pose_m2m_guards COMMAND test_relative_pose_m2m)
endif()
