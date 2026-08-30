if(NOT DEFINED BUILD_DIR OR NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "BUILD_DIR and TEST_ROOT are required")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
set(install_prefix "${TEST_ROOT}/install")
set(consumer_source "${TEST_ROOT}/consumer")
set(consumer_build "${TEST_ROOT}/consumer-build")
file(MAKE_DIRECTORY "${consumer_source}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${install_prefix}"
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_stdout
    ERROR_VARIABLE install_stderr)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR
        "installing DinoM2MLORansac and its bundled PoseLib failed:\n${install_stdout}\n${install_stderr}")
endif()

file(WRITE "${consumer_source}/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.16)\n"
    "project(DinoM2MInstallConsumer LANGUAGES CXX)\n"
    "find_package(DinoM2MLORansac CONFIG REQUIRED)\n"
    "add_executable(consumer main.cpp)\n"
    "target_link_libraries(consumer PRIVATE DinoM2M::loransac)\n")
file(WRITE "${consumer_source}/main.cpp"
    "#include <m2m_loransac/relative_pose_m2m.h>\n"
    "int main() {\n"
    "    return loransac_app::is_m2m_ransac(loransac_app::RansacMode::HCM) ? 0 : 1;\n"
    "}\n")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${consumer_source}" -B "${consumer_build}"
            "-DCMAKE_PREFIX_PATH=${install_prefix}"
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
        "installed-package consumer configure failed:\n${configure_stdout}\n${configure_stderr}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}"
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_stdout
    ERROR_VARIABLE build_stderr)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR
        "installed-package consumer build failed:\n${build_stdout}\n${build_stderr}")
endif()
