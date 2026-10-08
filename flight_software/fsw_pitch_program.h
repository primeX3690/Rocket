/* fsw_pitch_program.h -- open-loop pitch program for the first stage (the "pitch table" real launchers
 * fly until closed-loop guidance takes over). Piecewise-linear pitch-from-vertical vs time since liftoff,
 * held at the last value after the table ends. Pure functions, no state: the table is validated once
 * (finite, strictly increasing times, bounded slope) so a corrupt table is refused rather than flown. */
#ifndef FSW_PITCH_PROGRAM_H
#define FSW_PITCH_PROGRAM_H

#define FSW_PITCH_PROGRAM_MAX 16

typedef struct {
    float t[FSW_PITCH_PROGRAM_MAX];     /* s since liftoff, strictly increasing */
    float phi[FSW_PITCH_PROGRAM_MAX];   /* rad from vertical                    */
    int   n;                            /* 0 = no program                       */
} fsw_pitch_program_t;

/* Returns 0 on success. Rejects n<1 or >MAX, non-finite values, non-increasing times, |phi| > pi, and any
 * segment steeper than max_rate_rps (the attitude loop could not follow it; pass the core's slew limit). */
int   fsw_pitch_program_set(fsw_pitch_program_t *p, const float *t, const float *phi, int n, float max_rate_rps);
float fsw_pitch_program_eval(const fsw_pitch_program_t *p, float t_since_liftoff_s);

#endif
