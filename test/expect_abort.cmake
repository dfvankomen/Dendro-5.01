# Runs CMD (a ;-separated list) and passes only if it exits nonzero and its
# output matches EXPECT. CTest cannot check both: PASS_REGULAR_EXPRESSION
# ignores the exit code. Usage:
#   cmake -DCMD=<list> -DEXPECT=<regex> -P expect_abort.cmake
execute_process(COMMAND ${CMD} RESULT_VARIABLE rc OUTPUT_VARIABLE out
                ERROR_VARIABLE out TIMEOUT 120)
message("${out}")
if(rc EQUAL 0)
  message(FATAL_ERROR "expected a nonzero exit, got 0")
endif()
if(NOT out MATCHES "${EXPECT}")
  message(FATAL_ERROR "exit ${rc}, but no output matched: ${EXPECT}")
endif()
message("exit ${rc} with the expected abort message")
