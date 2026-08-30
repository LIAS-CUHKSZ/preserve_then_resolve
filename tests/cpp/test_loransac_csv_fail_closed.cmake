if(NOT DEFINED RUNNER_EXE OR NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "RUNNER_EXE and TEST_ROOT are required")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/dataset/method")
set(match_header "left_idx,right_idx,x1,y1,x2,y2,similarity,k_first\n")
string(CONCAT valid_rows
    "0,0,100,100,110,100,0.9,1\n"
    "1,1,180,140,195,140,0.8,1\n"
    "2,2,260,200,280,200,0.7,1\n"
    "3,3,340,260,365,260,0.6,1\n"
    "4,4,420,320,450,320,0.5,1\n")
file(WRITE "${TEST_ROOT}/dataset/method/matching_0.csv" "${match_header}${valid_rows}")
file(WRITE "${TEST_ROOT}/dataset/method/matching_1.csv"
    "${match_header}0,0,100\n${valid_rows}")
file(WRITE "${TEST_ROOT}/dataset/method/matching_2.csv"
    "${match_header}0,0,bad,100,110,100,0.9,1\n${valid_rows}")
file(WRITE "${TEST_ROOT}/dataset/method/matching_3.csv"
    "${match_header}0,0,100,nan,110,100,0.9,1\n${valid_rows}")
file(WRITE "${TEST_ROOT}/dataset/method/matching_4.csv"
    "${match_header}0,0,100,100,110,100,bad,1\n${valid_rows}")
file(WRITE "${TEST_ROOT}/dataset/method/matching_5.csv"
    "${match_header}0,0,100,100,110,100,inf,1\n${valid_rows}")
file(WRITE "${TEST_ROOT}/dataset/method/matching_6.csv"
    "${match_header}0,0,100junk,100,110,100,0.9,1\n${valid_rows}")
file(WRITE "${TEST_ROOT}/dataset/method/matching_7.csv"
    "${match_header}0,0,100,100,110,100,0.9,1,extra\n${valid_rows}")

set(pose_csv
    "pair_idx,qw,qx,qy,qz,tx,ty,tz,fx1,fy1,cx1,cy1,fx2,fy2,cx2,cy2\n")
foreach(pair_idx RANGE 0 7)
    string(APPEND pose_csv
        "${pair_idx},1,0,0,0,1,0,0,800,800,320,240,800,800,320,240\n")
endforeach()
file(WRITE "${TEST_ROOT}/dataset/pose_intrinsics.csv" "${pose_csv}")
file(SHA256 "${TEST_ROOT}/dataset/pose_intrinsics.csv" pose_sha256)
file(WRITE "${TEST_ROOT}/dataset/pose_intrinsics_manifest.json"
    "{\"pair_file_sha256\":\"pairs\",\"pair_identity_sha256\":\"identity\","
    "\"pair_count\":8,\"pose_intrinsics_sha256\":\"${pose_sha256}\"}\n")
file(WRITE "${TEST_ROOT}/dataset/method/association_manifest.json"
    "{\"pair_file_sha256\":\"pairs\",\"pair_identity_sha256\":\"identity\","
    "\"pair_count\":8}\n")
file(WRITE "${TEST_ROOT}/runner.cfg"
    "matching_result_root=.\n"
    "datasets=dataset\n"
    "method=method\n"
    "ransac_mode=HCM\n"
    "q_ub=0.3\n"
    "m2m_delta=0.01\n"
    "init_with_gt=true\n"
    "min_iterations=0\n"
    "max_iterations=0\n"
    "allow_unbound_pose=false\n"
    "output_csv=csv_guard.csv\n")

execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/runner.cfg"
    RESULT_VARIABLE runner_result
    OUTPUT_VARIABLE runner_stdout
    ERROR_VARIABLE runner_stderr)
if(NOT runner_result EQUAL 0)
    message(FATAL_ERROR "runner failed instead of isolating malformed pairs: ${runner_stderr}")
endif()

set(result_csv "${TEST_ROOT}/dataset/method/wtgt_HCM_csv_guard_q_ub_0.30.csv")
file(STRINGS "${result_csv}" rows)
list(LENGTH rows row_count)
if(NOT row_count EQUAL 9)
    message(FATAL_ERROR "expected one header plus eight pair rows, got ${row_count}")
endif()
list(GET rows 1 valid_row)
if(NOT valid_row MATCHES "^0,success,,5,")
    message(FATAL_ERROR "legal CSV behavior changed: ${valid_row}")
endif()

set(expected_fragments
    "Invalid column count on line 2"
    "Invalid x1 on line 2"
    "Invalid y1 on line 2"
    "Invalid similarity on line 2"
    "Invalid similarity on line 2"
    "Invalid x1 on line 2"
    "Invalid column count on line 2")
# CMake list positions are one greater than pair IDs because the header is row 0.
foreach(pair_idx RANGE 1 7)
    math(EXPR row_position "${pair_idx} + 1")
    math(EXPR fragment_position "${pair_idx} - 1")
    list(GET rows ${row_position} malformed_row)
    list(GET expected_fragments ${fragment_position} expected_fragment)
    if(NOT malformed_row MATCHES
       "^${pair_idx},skipped,invalid_match_csv: ${expected_fragment}.*matching_${pair_idx}\\.csv[^,]*,0,")
        message(FATAL_ERROR
            "pair ${pair_idx} did not fail closed with source context: ${malformed_row}")
    endif()
endforeach()

if(NOT runner_stdout MATCHES "Total processed 1 pair files \\(7 skipped\\)")
    message(FATAL_ERROR "runner did not continue after pair-local CSV errors: ${runner_stdout}")
endif()
