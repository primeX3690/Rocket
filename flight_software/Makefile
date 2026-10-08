# Host build + self-test for the flight-software layer (needs only gcc/clang).
CC      ?= gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror
SRCS     = fsw_math.c fsw_pitch_program.c fsw_frames.c fsw_peg.c fsw_guidance.c fsw_eskf.c fsw_crc.c fsw_telemetry.c fsw_fdir.c fsw_state.c fsw_watchdog.c fsw_core.c
EMB      = ../embedded/pid_controller.c   # reused TVC PID/actuator/plant (built without -Werror)

.PHONY: test clean
test: build/fsw_selftest
	./build/fsw_selftest

build/fsw_selftest: $(SRCS) test/fsw_selftest.c $(EMB)
	mkdir -p build
	$(CC) $(CFLAGS) -c $(SRCS) test/fsw_selftest.c
	$(CC) -std=c99 -O2 -c $(EMB) -o pid_controller.o
	$(CC) -o $@ *.o -lm   # libm only for the test (reference math); flight objects need none
	rm -f *.o

clean:
	rm -rf build *.o
