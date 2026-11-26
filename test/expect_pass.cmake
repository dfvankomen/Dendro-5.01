# Runs CMD (a ;-separated list) and passes only if it exits 0 and its output
# matches EXPECT. PASS_REGULAR_EXPRESSION alone ignores the exit code, so a
# crash after the PASS line would pass. Usage:
#   cmake -DCMD=<list> -DEXPECT=<regex> -P expect_pass.cmake
execute_process(COMMAND ${CMD} RESULT_VARIABLE rc OUTPUT_VARIABLE out
                ERROR_VARIABLE out TIMEOUT 240)
message("${out}")
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "exit ${rc}, expected 0")
endif()
if(NOT out MATCHES "${EXPECT}")
  message(FATAL_ERROR "exit 0, but no output matched: ${EXPECT}")
endif()
