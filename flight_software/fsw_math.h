/* fsw_math.h -- the handful of math functions the flight code needs,
 * implemented without libm so the object files carry no libm dependency
 * (checked by tests/test_eskf_c_equivalence.py via `nm -u`). Accuracy is
 * stated per function and verified against libm in the host self-test.
 * IEEE-754 single precision; do not build with -ffast-math. */
#ifndef FSW_MATH_H
#define FSW_MATH_H

#define FSW_PI_F 3.14159265358979323846f

float fsw_sqrtf(float x);            /* rel. err ~1e-7; x<=0 or NaN -> 0 (NaN in -> NaN out) */
float fsw_sinf(float x);             /* abs err ~3e-6 on any finite |x| < ~1e4              */
float fsw_cosf(float x);             /* abs err ~3e-6                                       */
float fsw_atanf(float x);            /* abs err ~1e-5 rad                                   */
float fsw_atan2f(float y, float x);  /* abs err ~1e-5 rad; atan2(0,0) = 0                   */
float fsw_asinf(float x);            /* abs err ~2e-5 rad (worst near |x|=1); x clamped     */
double fsw_sqrt_d(double x);         /* full double precision (float seed + 2 Newton steps), no libm; x<=0 -> 0 */
float fsw_absf(float x);
int   fsw_isfinitef(float x);

#endif
