if(NOT DEFINED GMS_EXE OR NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "GMS_EXE and TEST_ROOT are required")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
set(valid_dir "${TEST_ROOT}/valid/matches")
file(MAKE_DIRECTORY "${valid_dir}")
set(expected_header "left_idx,right_idx,x1,y1,x2,y2,similarity,k\n")
file(WRITE "${valid_dir}/matching_0.csv" "${expected_header}")
file(WRITE "${valid_dir}/association_manifest.json"
    "{\"pair_file_sha256\":\"pairs\",\"pair_identity_sha256\":\"identity\"}\n")
file(WRITE "${valid_dir}/matching_1.csv" "${expected_header}"
    "0,0,100,100,110,100,0.90,5\n"
    "1,1,101,101,111,101,0.89,5\n"
    "2,2,102,102,112,102,0.88,5\n"
    "3,3,103,103,113,103,0.87,5\n"
    "4,4,104,104,114,104,0.86,5\n"
    "5,5,105,105,115,105,0.85,5\n"
    "6,6,106,106,116,106,0.84,5\n"
    "7,7,107,107,117,107,0.83,5\n"
    "8,8,108,108,118,108,0.82,5\n"
    "9,9,109,109,119,109,0.81,5\n")
file(WRITE "${TEST_ROOT}/pose_intrinsics.csv"
    "pair_idx,cx1,cy1,cx2,cy2\n"
    "0,500,500,500,500\n"
    "1,500,500,500,500\n")

execute_process(
    COMMAND "${GMS_EXE}" --root "${valid_dir}"
    RESULT_VARIABLE valid_result
    OUTPUT_VARIABLE valid_stdout
    ERROR_VARIABLE valid_stderr)
if(NOT valid_result EQUAL 0)
    message(FATAL_ERROR "header-only CSV failed: ${valid_stderr}")
endif()

set(filtered_csv "${TEST_ROOT}/valid/matches_GMSm2m_ThrFact_6.0_Gridsz_20/matching_0.csv")
if(NOT EXISTS "${filtered_csv}")
    message(FATAL_ERROR "header-only output was not created")
endif()
file(READ "${filtered_csv}" actual_output)
if(NOT actual_output STREQUAL expected_header)
    message(FATAL_ERROR "header-only output changed schema: '${actual_output}'")
endif()

set(nonempty_csv "${TEST_ROOT}/valid/matches_GMSm2m_ThrFact_6.0_Gridsz_20/matching_1.csv")
if(NOT EXISTS "${nonempty_csv}")
    message(FATAL_ERROR "nonempty GMS output was not created")
endif()
set(filtered_manifest "${TEST_ROOT}/valid/matches_GMSm2m_ThrFact_6.0_Gridsz_20/association_manifest.json")
if(NOT EXISTS "${filtered_manifest}")
    message(FATAL_ERROR "association manifest was not propagated by GMS")
endif()
file(READ "${valid_dir}/association_manifest.json" input_manifest)
file(READ "${filtered_manifest}" output_manifest)
if(NOT input_manifest STREQUAL output_manifest)
    message(FATAL_ERROR "GMS changed the association manifest")
endif()
file(STRINGS "${nonempty_csv}" nonempty_rows)
list(LENGTH nonempty_rows nonempty_row_count)
if(NOT nonempty_row_count EQUAL 11)
    message(FATAL_ERROR "expected all 10 coherent associations to survive, got ${nonempty_row_count} rows")
endif()
list(GET nonempty_rows 1 first_nonempty_row)
if(NOT first_nonempty_row STREQUAL "0,0,100,100,110,100,0.90,5")
    message(FATAL_ERROR "nonempty GMS output changed the association payload: ${first_nonempty_row}")
endif()

set(invalid_dir "${TEST_ROOT}/invalid/matches")
file(MAKE_DIRECTORY "${invalid_dir}")
file(WRITE "${invalid_dir}/matching_0.csv" "left_idx,right_idx,x1,y1\n")
execute_process(
    COMMAND "${GMS_EXE}" --root "${invalid_dir}"
    RESULT_VARIABLE invalid_result
    OUTPUT_QUIET
    ERROR_VARIABLE invalid_stderr)
if(invalid_result EQUAL 0)
    message(FATAL_ERROR "invalid CSV schema was accepted")
endif()
if(NOT invalid_stderr MATCHES "x1,y1,x2,y2")
    message(FATAL_ERROR "invalid CSV error was not actionable: ${invalid_stderr}")
endif()
