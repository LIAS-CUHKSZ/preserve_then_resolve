if(NOT DEFINED RUNNER_EXE OR NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "RUNNER_EXE and TEST_ROOT are required")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/dataset/method")
file(WRITE "${TEST_ROOT}/dataset/method/matching_0.csv"
    "left_idx,right_idx,x1,y1,x2,y2,similarity,k\n")
file(WRITE "${TEST_ROOT}/dataset/method/matching_1.csv"
    "left_idx,right_idx,x1,y1,x2,y2,similarity,k\n"
    "0,0,100,100,110,100,0.9,5\n"
    "0,1,180,140,195,140,0.9,5\n"
    "0,2,260,200,280,200,0.9,5\n"
    "0,3,340,260,365,260,0.9,5\n"
    "0,4,420,320,450,320,0.9,5\n")
file(WRITE "${TEST_ROOT}/dataset/method/matching_2.csv"
    "left_idx,right_idx,x1,y1,x2,y2,similarity,k\n"
    "-1,0,100,100,110,100,0.9,5\n"
    "1,1,180,140,195,140,0.9,5\n"
    "2,2,260,200,280,200,0.9,5\n"
    "3,3,340,260,365,260,0.9,5\n"
    "4,4,420,320,450,320,0.9,5\n")
file(WRITE "${TEST_ROOT}/dataset/method/matching_3.csv"
    "left_idx,right_idx,x1,y1,x2,y2,similarity,k\n"
    "0,0,100,100,110,100,0.9,5\n"
    "1,1,180,140,195,140,0.9,5\n"
    "2,2,260,200,280,200,0.9,5\n"
    "3,3,340,260,365,260,0.9,5\n"
    "4,4,420,320,450,320,0.9,5\n")
file(WRITE "${TEST_ROOT}/dataset/method/matching_4.csv"
    "left_idx,right_idx,x1,y1,x2,y2,similarity,k\n"
    "bad,0,100,100,110,100,0.9,5\n"
    "1,1,180,140,195,140,0.9,5\n"
    "2,2,260,200,280,200,0.9,5\n"
    "3,3,340,260,365,260,0.9,5\n"
    "4,4,420,320,450,320,0.9,5\n")
string(CONCAT pose_csv_contents
    "pair_idx,qw,qx,qy,qz,tx,ty,tz,fx1,fy1,cx1,cy1,fx2,fy2,cx2,cy2\n"
    "0,1,0,0,0,1,0,0,800,800,320,240,800,800,320,240\n"
    "1,1,0,0,0,1,0,0,800,800,320,240,800,800,320,240\n"
    "2,1,0,0,0,1,0,0,800,800,320,240,800,800,320,240\n"
    "3,1,0,0,0,1,0,0,800,800,320,240,800,800,320,240\n"
    "4,1,0,0,0,1,0,0,800,800,320,240,800,800,320,240\n")
file(WRITE "${TEST_ROOT}/dataset/pose_intrinsics.csv" "${pose_csv_contents}")
file(SHA256 "${TEST_ROOT}/dataset/pose_intrinsics.csv" pose_csv_sha256)
set(pose_manifest_contents
    "{\"pair_file_sha256\":\"pairs\",\"pair_identity_sha256\":\"identity\",\"pair_count\":5,\"pose_intrinsics_sha256\":\"${pose_csv_sha256}\"}\n")
set(association_manifest_contents
    "{\"pair_file_sha256\":\"pairs\",\"pair_identity_sha256\":\"identity\",\"pair_count\":5}\n")
file(WRITE "${TEST_ROOT}/dataset/pose_intrinsics_manifest.json" "${pose_manifest_contents}")
file(WRITE "${TEST_ROOT}/dataset/method/association_manifest.json" "${association_manifest_contents}")
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
    "output_csv=result.csv\n")

file(WRITE "${TEST_ROOT}/legacy_gt_mode.cfg"
    "matching_result_root=.\n"
    "datasets=dataset\n"
    "method=method\n"
    "gt_mode=inverse\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/legacy_gt_mode.cfg"
    RESULT_VARIABLE legacy_result
    OUTPUT_QUIET
    ERROR_VARIABLE legacy_stderr)
if(legacy_result EQUAL 0 OR NOT legacy_stderr MATCHES "gt_mode has been removed")
    message(FATAL_ERROR "legacy gt_mode was not rejected clearly: ${legacy_stderr}")
endif()

execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/runner.cfg"
    RESULT_VARIABLE runner_result
    OUTPUT_VARIABLE runner_stdout
    ERROR_VARIABLE runner_stderr)
if(NOT runner_result EQUAL 0)
    message(FATAL_ERROR "runner guard fixture failed: ${runner_stderr}")
endif()

set(result_csv "${TEST_ROOT}/dataset/method/wtgt_HCM_result_q_ub_0.30.csv")
if(NOT EXISTS "${result_csv}")
    message(FATAL_ERROR "runner did not create expected result CSV")
endif()
file(STRINGS "${result_csv}" rows)
list(LENGTH rows row_count)
if(NOT row_count EQUAL 6)
    message(FATAL_ERROR "expected header plus five result rows, got ${row_count} rows")
endif()
list(GET rows 0 header)
list(GET rows 1 empty_skipped)
list(GET rows 2 no_model_skipped)
list(GET rows 3 negative_id_skipped)
list(GET rows 4 success)
list(GET rows 5 malformed_id_skipped)
string(REGEX MATCHALL "," header_commas "${header}")
string(REGEX MATCHALL "," skipped_commas "${empty_skipped}")
list(LENGTH header_commas header_comma_count)
list(LENGTH skipped_commas skipped_comma_count)
if(NOT header_comma_count EQUAL 32 OR NOT skipped_comma_count EQUAL 32)
    message(FATAL_ERROR
        "result width mismatch: header=${header_comma_count}, skipped=${skipped_comma_count}")
endif()
if(header MATCHES "gt_mode")
    message(FATAL_ERROR "deprecated GT convention leaked into result schema: ${header}")
endif()
if(NOT empty_skipped MATCHES "^0,skipped,not_enough_matches,0,")
    message(FATAL_ERROR "unexpected empty-input row: ${empty_skipped}")
endif()
if(NOT no_model_skipped MATCHES "^1,skipped,proposal_graph_not_enough_o2o,5,")
    message(FATAL_ERROR "no-model input was not skipped: ${no_model_skipped}")
endif()
if(NOT negative_id_skipped MATCHES "^2,skipped,invalid_feature_ids,5,")
    message(FATAL_ERROR "negative feature IDs were not isolated to their pair: ${negative_id_skipped}")
endif()
if(NOT success MATCHES "^3,success,,5,")
    message(FATAL_ERROR "valid initial-model input did not succeed: ${success}")
endif()
if(NOT malformed_id_skipped MATCHES "^4,skipped,invalid_match_csv: Invalid feature ID left_idx")
    message(FATAL_ERROR "malformed feature ID was not isolated to its pair: ${malformed_id_skipped}")
endif()

# Controlled proposal sampling: pair 0 has only four E1 edges although its
# full E5 graph has ten, so the sampler/cardinality guard must reject it. Pair
# 1 has five E1 edges, must run exactly the fixed three iterations, and must
# retain all ten E5 rows as the scoring graph. Exercise both the existing `k`
# header and the canonical `k_first` spelling.
file(MAKE_DIRECTORY "${TEST_ROOT}/proposal_dataset/method")
file(WRITE "${TEST_ROOT}/proposal_dataset/method/matching_0.csv"
    "left_idx,right_idx,x1,y1,x2,y2,similarity,k\n"
    "0,0,100,100,110,100,0.9,1\n"
    "1,1,180,140,195,140,0.9,1\n"
    "2,2,260,200,280,200,0.9,1\n"
    "3,3,340,260,365,260,0.9,1\n"
    "4,4,420,320,450,320,0.9,5\n"
    "5,5,120,360,135,360,0.9,5\n"
    "6,6,200,380,220,380,0.9,5\n"
    "7,7,280,400,305,400,0.9,5\n"
    "8,8,360,420,390,420,0.9,5\n"
    "9,9,440,440,475,440,0.9,5\n")
file(WRITE "${TEST_ROOT}/proposal_dataset/method/matching_1.csv"
    "left_idx,right_idx,x1,y1,x2,y2,similarity,k_first\n"
    "0,0,100,100,110,100,0.9,1\n"
    "1,1,180,140,195,140,0.9,1\n"
    "2,2,260,200,280,200,0.9,1\n"
    "3,3,340,260,365,260,0.9,1\n"
    "4,4,420,320,450,320,0.9,1\n"
    "5,5,120,360,135,360,0.9,5\n"
    "6,6,200,380,220,380,0.9,5\n"
    "7,7,280,400,305,400,0.9,5\n"
    "8,8,360,420,390,420,0.9,5\n"
    "9,9,440,440,475,440,0.9,5\n")
string(CONCAT proposal_pose_contents
    "pair_idx,qw,qx,qy,qz,tx,ty,tz,fx1,fy1,cx1,cy1,fx2,fy2,cx2,cy2\n"
    "0,1,0,0,0,1,0,0,800,800,320,240,800,800,320,240\n"
    "1,1,0,0,0,1,0,0,800,800,320,240,800,800,320,240\n")
file(WRITE "${TEST_ROOT}/proposal_dataset/pose_intrinsics.csv" "${proposal_pose_contents}")
file(SHA256 "${TEST_ROOT}/proposal_dataset/pose_intrinsics.csv" proposal_pose_sha256)
file(WRITE "${TEST_ROOT}/proposal_dataset/pose_intrinsics_manifest.json"
    "{\"pair_file_sha256\":\"proposal-pairs\",\"pair_identity_sha256\":\"proposal-identity\","
    "\"pair_count\":2,\"pose_intrinsics_sha256\":\"${proposal_pose_sha256}\"}\n")
file(WRITE "${TEST_ROOT}/proposal_dataset/method/association_manifest.json"
    "{\"pair_file_sha256\":\"proposal-pairs\",\"pair_identity_sha256\":\"proposal-identity\","
    "\"pair_count\":2}\n")
file(WRITE "${TEST_ROOT}/proposal_fixed.cfg"
    "matching_result_root=.\n"
    "datasets=proposal_dataset\n"
    "method=method\n"
    "ransac_mode=HCM\n"
    "proposal_max_k=1\n"
    "q_ub=0.3\n"
    "m2m_delta=0.01\n"
    "max_matching_num=0\n"
    "similarity_threshold=0\n"
    "init_with_gt=true\n"
    "min_iterations=3\n"
    "max_iterations=3\n"
    "allow_unbound_pose=false\n"
    "output_csv=proposal_fixed.csv\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/proposal_fixed.cfg"
    RESULT_VARIABLE proposal_result
    OUTPUT_VARIABLE proposal_stdout
    ERROR_VARIABLE proposal_stderr)
if(NOT proposal_result EQUAL 0)
    message(FATAL_ERROR "proposal-source fixture failed: ${proposal_stderr}")
endif()
set(proposal_result_csv
    "${TEST_ROOT}/proposal_dataset/method/wtgt_HCM_proposal_fixed_proposal_k_1_q_ub_0.30.csv")
file(STRINGS "${proposal_result_csv}" proposal_rows)
list(LENGTH proposal_rows proposal_row_count)
if(NOT proposal_row_count EQUAL 3)
    message(FATAL_ERROR "proposal fixture expected header plus two rows")
endif()
list(GET proposal_rows 1 low_cardinality_row)
list(GET proposal_rows 2 fixed_row)
if(NOT low_cardinality_row MATCHES "^0,skipped,proposal_graph_not_enough_o2o,10,")
    message(FATAL_ERROR "proposal sampler did not reject low-cardinality E1: ${low_cardinality_row}")
endif()
if(NOT fixed_row MATCHES "^1,success,,10,[0-9]+,[^,]+,3,")
    message(FATAL_ERROR "fixed proposal run did not retain E5 and execute 3 iterations: ${fixed_row}")
endif()

# The GT hypothesis probe is a separate sidecar on an otherwise normal
# HCM+MCM run. Pair 0 remains a normal proposal failure; pair 1 must expose a
# full capacity-one raw-seed pool and one read-only admission decision.
file(WRITE "${TEST_ROOT}/proposal_probe.cfg"
    "matching_result_root=.\n"
    "datasets=proposal_dataset\n"
    "method=method\n"
    "ransac_mode=HCM_MC\n"
    "top_n_candidates=1\n"
    "write_gt_hypothesis_probe=true\n"
    "proposal_max_k=1\n"
    "q_ub=0.3\n"
    "m2m_delta=0.01\n"
    "max_matching_num=0\n"
    "similarity_threshold=0\n"
    "init_with_gt=false\n"
    "ransac_times=1\n"
    "min_iterations=3\n"
    "max_iterations=3\n"
    "allow_unbound_pose=false\n"
    "output_csv=proposal_probe.csv\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/proposal_probe.cfg"
    RESULT_VARIABLE probe_result
    OUTPUT_VARIABLE probe_stdout
    ERROR_VARIABLE probe_stderr)
if(NOT probe_result EQUAL 0)
    message(FATAL_ERROR "ground-truth hypothesis probe fixture failed: ${probe_stderr}")
endif()
set(probe_pose_csv
    "${TEST_ROOT}/proposal_dataset/method/HCM_MC_proposal_probe_proposal_k_1_q_ub_0.30.csv")
set(probe_sidecar_csv
    "${TEST_ROOT}/proposal_dataset/method/HCM_MC_proposal_probe_proposal_k_1_q_ub_0.30_gt_hypothesis_probe.csv")
if(NOT EXISTS "${probe_pose_csv}" OR NOT EXISTS "${probe_sidecar_csv}")
    message(FATAL_ERROR "probe run did not create both the normal result and sidecar")
endif()
file(STRINGS "${probe_sidecar_csv}" probe_rows)
list(LENGTH probe_rows probe_row_count)
if(NOT probe_row_count EQUAL 3)
    message(FATAL_ERROR "probe sidecar expected header plus two rows")
endif()
list(GET probe_rows 0 probe_header)
list(GET probe_rows 1 probe_skipped)
list(GET probe_rows 2 probe_success)
if(NOT probe_header STREQUAL
   "pair_idx,status,error_message,pool_size,pool_capacity,pool_full,gt_hcm_score,pool_cutoff_hcm_score,gt_edge_inliers,would_enter_top_n")
    message(FATAL_ERROR "unexpected GT hypothesis probe schema: ${probe_header}")
endif()
if(NOT probe_skipped MATCHES "^0,skipped,proposal_graph_not_enough_o2o,")
    message(FATAL_ERROR "proposal failure was not mirrored in probe sidecar: ${probe_skipped}")
endif()
if(NOT probe_success MATCHES "^1,success,,1,1,1,-?[0-9.]+,-?[0-9.]+,[0-9]+,[01]$")
    message(FATAL_ERROR "unexpected successful probe row: ${probe_success}")
endif()

# Enabling proposal sampling with adaptive bounds is intentionally rejected.
file(WRITE "${TEST_ROOT}/proposal_adaptive.cfg"
    "matching_result_root=.\n"
    "datasets=proposal_dataset\n"
    "method=method\n"
    "ransac_mode=HCM\n"
    "proposal_max_k=1\n"
    "min_iterations=0\n"
    "max_iterations=3\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/proposal_adaptive.cfg"
    RESULT_VARIABLE adaptive_result
    OUTPUT_QUIET
    ERROR_VARIABLE adaptive_stderr)
if(adaptive_result EQUAL 0 OR NOT adaptive_stderr MATCHES "fixed iteration budget")
    message(FATAL_ERROR "adaptive proposal budget was not rejected: ${adaptive_stderr}")
endif()

# Missing rank metadata must never fall back to full-graph proposals.
file(MAKE_DIRECTORY "${TEST_ROOT}/missing_rank_dataset/method")
file(WRITE "${TEST_ROOT}/missing_rank_dataset/method/matching_0.csv"
    "left_idx,right_idx,x1,y1,x2,y2,similarity\n"
    "0,0,100,100,110,100,0.9\n"
    "1,1,180,140,195,140,0.9\n"
    "2,2,260,200,280,200,0.9\n"
    "3,3,340,260,365,260,0.9\n"
    "4,4,420,320,450,320,0.9\n")
file(WRITE "${TEST_ROOT}/missing_rank_dataset/pose_intrinsics.csv"
    "pair_idx,qw,qx,qy,qz,tx,ty,tz,fx1,fy1,cx1,cy1,fx2,fy2,cx2,cy2\n"
    "0,1,0,0,0,1,0,0,800,800,320,240,800,800,320,240\n")
file(SHA256 "${TEST_ROOT}/missing_rank_dataset/pose_intrinsics.csv" missing_rank_pose_sha256)
file(WRITE "${TEST_ROOT}/missing_rank_dataset/pose_intrinsics_manifest.json"
    "{\"pair_file_sha256\":\"missing-rank\",\"pair_identity_sha256\":\"missing-rank\","
    "\"pair_count\":1,\"pose_intrinsics_sha256\":\"${missing_rank_pose_sha256}\"}\n")
file(WRITE "${TEST_ROOT}/missing_rank_dataset/method/association_manifest.json"
    "{\"pair_file_sha256\":\"missing-rank\",\"pair_identity_sha256\":\"missing-rank\","
    "\"pair_count\":1}\n")
file(WRITE "${TEST_ROOT}/missing_rank.cfg"
    "matching_result_root=.\n"
    "datasets=missing_rank_dataset\n"
    "method=method\n"
    "ransac_mode=HCM\n"
    "proposal_max_k=1\n"
    "min_iterations=0\n"
    "max_iterations=0\n"
    "output_csv=missing_rank.csv\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/missing_rank.cfg"
    RESULT_VARIABLE missing_rank_result
    OUTPUT_QUIET
    ERROR_VARIABLE missing_rank_stderr)
if(NOT missing_rank_result EQUAL 0)
    message(FATAL_ERROR "missing-rank fixture runner failed unexpectedly: ${missing_rank_stderr}")
endif()
set(missing_rank_csv
    "${TEST_ROOT}/missing_rank_dataset/method/HCM_missing_rank_proposal_k_1_q_ub_0.30.csv")
file(STRINGS "${missing_rank_csv}" missing_rank_rows)
list(GET missing_rank_rows 1 missing_rank_row)
if(NOT missing_rank_row MATCHES
   "^0,skipped,invalid_match_csv: proposal_max_k requires a k_first or k column")
    message(FATAL_ERROR "missing proposal rank did not fail closed: ${missing_rank_row}")
endif()

# min/max ordering is now validated explicitly instead of relying on PoseLib.
file(WRITE "${TEST_ROOT}/invalid_iteration_order.cfg"
    "matching_result_root=.\n"
    "datasets=dataset\n"
    "method=method\n"
    "min_iterations=4\n"
    "max_iterations=3\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/invalid_iteration_order.cfg"
    RESULT_VARIABLE iteration_order_result
    OUTPUT_QUIET
    ERROR_VARIABLE iteration_order_stderr)
if(iteration_order_result EQUAL 0 OR NOT iteration_order_stderr MATCHES "must not exceed")
    message(FATAL_ERROR "invalid iteration ordering was not rejected: ${iteration_order_stderr}")
endif()

# Pair-bound pose metadata must reject associations from another split before
# reading or resuming estimator results.
file(WRITE "${TEST_ROOT}/dataset/method/association_manifest.json"
    "{\"pair_file_sha256\":\"other\",\"pair_identity_sha256\":\"identity\",\"pair_count\":5}\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/runner.cfg"
    RESULT_VARIABLE provenance_result
    OUTPUT_QUIET
    ERROR_VARIABLE provenance_stderr)
if(provenance_result EQUAL 0 OR NOT provenance_stderr MATCHES "Pair provenance mismatch")
    message(FATAL_ERROR "pair provenance mismatch was not rejected: ${provenance_stderr}")
endif()
file(WRITE "${TEST_ROOT}/dataset/method/association_manifest.json" "${association_manifest_contents}")

# Current runs require the pose sidecar. Historical trees must opt in explicitly.
file(REMOVE "${TEST_ROOT}/dataset/pose_intrinsics_manifest.json")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/runner.cfg"
    RESULT_VARIABLE missing_manifest_result
    OUTPUT_QUIET
    ERROR_VARIABLE missing_manifest_stderr)
if(missing_manifest_result EQUAL 0 OR NOT missing_manifest_stderr MATCHES "manifest is required")
    message(FATAL_ERROR "missing pose manifest was not rejected: ${missing_manifest_stderr}")
endif()
file(WRITE "${TEST_ROOT}/historical_unbound.cfg"
    "matching_result_root=.\n"
    "datasets=dataset\n"
    "method=method\n"
    "ransac_mode=HCM\n"
    "q_ub=0.3\n"
    "m2m_delta=0.01\n"
    "init_with_gt=true\n"
    "min_iterations=0\n"
    "max_iterations=0\n"
    "allow_unbound_pose=true\n"
    "output_csv=historical_unbound.csv\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/historical_unbound.cfg"
    RESULT_VARIABLE unbound_result
    OUTPUT_QUIET
    ERROR_VARIABLE unbound_stderr)
if(NOT unbound_result EQUAL 0)
    message(FATAL_ERROR "explicit historical pose opt-in failed: ${unbound_stderr}")
endif()
file(WRITE "${TEST_ROOT}/dataset/pose_intrinsics_manifest.json" "${pose_manifest_contents}")

# The sidecar must bind the actual pose bytes, not only the pair-list hashes.
file(APPEND "${TEST_ROOT}/dataset/pose_intrinsics.csv" "\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/runner.cfg"
    RESULT_VARIABLE pose_hash_result
    OUTPUT_QUIET
    ERROR_VARIABLE pose_hash_stderr)
if(pose_hash_result EQUAL 0 OR NOT pose_hash_stderr MATCHES "pose_intrinsics_sha256 mismatch")
    message(FATAL_ERROR "modified pose CSV was not rejected: ${pose_hash_stderr}")
endif()
file(WRITE "${TEST_ROOT}/dataset/pose_intrinsics.csv" "${pose_csv_contents}")

# Counts and exact pair-ID sets must agree across pose and association files.
file(WRITE "${TEST_ROOT}/dataset/pose_intrinsics_manifest.json"
    "{\"pair_file_sha256\":\"pairs\",\"pair_identity_sha256\":\"identity\",\"pair_count\":4,\"pose_intrinsics_sha256\":\"${pose_csv_sha256}\"}\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/runner.cfg"
    RESULT_VARIABLE pair_count_result
    OUTPUT_QUIET
    ERROR_VARIABLE pair_count_stderr)
if(pair_count_result EQUAL 0 OR NOT pair_count_stderr MATCHES "pair_count mismatch")
    message(FATAL_ERROR "manifest pair_count mismatch was not rejected: ${pair_count_stderr}")
endif()
file(WRITE "${TEST_ROOT}/dataset/pose_intrinsics_manifest.json" "${pose_manifest_contents}")
file(RENAME
    "${TEST_ROOT}/dataset/method/matching_4.csv"
    "${TEST_ROOT}/dataset/method/matching_5.csv")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/runner.cfg"
    RESULT_VARIABLE pair_ids_result
    OUTPUT_QUIET
    ERROR_VARIABLE pair_ids_stderr)
if(pair_ids_result EQUAL 0 OR NOT pair_ids_stderr MATCHES "pair IDs do not match")
    message(FATAL_ERROR "matching/pose pair-ID mismatch was not rejected: ${pair_ids_stderr}")
endif()
file(RENAME
    "${TEST_ROOT}/dataset/method/matching_5.csv"
    "${TEST_ROOT}/dataset/method/matching_4.csv")

# Resuming into a pre-standardization result must fail instead of appending
# 33-column rows beneath a legacy 34-column header.
set(legacy_result_csv "${TEST_ROOT}/dataset/method/wtgt_HCM_legacy_resume_q_ub_0.30.csv")
file(WRITE "${legacy_result_csv}"
    "pair_idx,status,error_message,num_matches,num_inliers,inlier_ratio,iterations,refinements,model_score,"
    "ransac_times,best_run_idx,q_w,q_x,q_y,q_z,t_x,t_y,t_z,gt_mode_used,rotation_error_deg,"
    "translation_error_deg,running_time_s,solve_ms,score_ms,score_calls,score_us_per_eval,refine_ms,"
    "refine_calls,rank_score_ms,rank_score_calls,ransac_total_ms,pool_insert_attempts,pool_dup_hits,"
    "pool_unique_basins\n"
    "0,success\n")
file(WRITE "${TEST_ROOT}/legacy_resume.cfg"
    "matching_result_root=.\n"
    "datasets=dataset\n"
    "method=method\n"
    "ransac_mode=HCM\n"
    "q_ub=0.3\n"
    "m2m_delta=0.01\n"
    "init_with_gt=true\n"
    "output_csv=legacy_resume.csv\n")
execute_process(
    COMMAND "${RUNNER_EXE}" "${TEST_ROOT}/legacy_resume.cfg"
    RESULT_VARIABLE resume_result
    OUTPUT_QUIET
    ERROR_VARIABLE resume_stderr)
if(resume_result EQUAL 0 OR NOT resume_stderr MATCHES "legacy gt_mode_used schema")
    message(FATAL_ERROR "legacy result schema was not rejected before resume: ${resume_stderr}")
endif()
