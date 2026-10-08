#include "fsw_math.h"
#include <stdint.h>
#include <string.h>

float fsw_absf(float x) { return (x < 0.0f) ? -x : x; }

int fsw_isfinitef(float x)
{
    return (x == x) && (x <= 3.4e38f) && (x >= -3.4e38f);
}

float fsw_sqrtf(float x)
{
    uint32_t i;
    float y;
    int k;
    if (x != x) {
        return x;                      /* propagate NaN */
    }
    if (x <= 0.0f) {
        return 0.0f;
    }
    if (x > 3.4e38f) {
        return x;                      /* +Inf */
    }
    memcpy(&i, &x, sizeof(i));
    i = 0x1FBD1DF5u + (i >> 1);        /* ~3.5 % initial guess */
    memcpy(&y, &i, sizeof(y));
    for (k = 0; k < 3; k++) {
        y = 0.5f * (y + x / y);        /* quadratic convergence */
    }
    return y;
}

/* Even degree-10 minimax polynomial for cos on [-pi, pi]
 * (same coefficients as embedded/pid_controller.c, max err ~2.4e-6). */
float fsw_cosf(float x)
{
    float x2;
    while (x > FSW_PI_F)  { x -= 2.0f * FSW_PI_F; }
    while (x < -FSW_PI_F) { x += 2.0f * FSW_PI_F; }
    x2 = x * x;
    return 9.99999442e-01f
         + x2 * (-4.99995572e-01f
         + x2 * (4.16610230e-02f
         + x2 * (-1.38627136e-03f
         + x2 * (2.42527302e-05f
         + x2 * (-2.21917677e-07f)))));
}

float fsw_sinf(float x)
{
    return fsw_cosf(x - 0.5f * FSW_PI_F);
}

float fsw_atanf(float x)
{
    /* |x| <= 1: odd minimax polynomial; |x| > 1: atan(x) = sign*pi/2 - atan(1/x). */
    float ax = fsw_absf(x);
    float r, z, z2;
    int inv = (ax > 1.0f);
    z = inv ? (1.0f / ax) : ax;
    z2 = z * z;
    r = z * (0.9998660f + z2 * (-0.3302995f + z2 * (0.1801410f + z2 * (-0.0851330f + z2 * 0.0208351f))));
    if (inv) {
        r = 0.5f * FSW_PI_F - r;
    }
    return (x < 0.0f) ? -r : r;
}

float fsw_atan2f(float y, float x)
{
    if (x == 0.0f) {
        if (y > 0.0f) { return 0.5f * FSW_PI_F; }
        if (y < 0.0f) { return -0.5f * FSW_PI_F; }
        return 0.0f;
    }
    if (fsw_absf(x) >= fsw_absf(y)) {
        float a = fsw_atanf(y / x);
        if (x > 0.0f) { return a; }
        return (y >= 0.0f) ? (a + FSW_PI_F) : (a - FSW_PI_F);
    } else {
        float a = fsw_atanf(x / y);          /* atan2(y,x) = +-pi/2 - atan(x/y) */
        return (y > 0.0f) ? (0.5f * FSW_PI_F - a) : (-0.5f * FSW_PI_F - a);
    }
}

float fsw_asinf(float x)
{
    if (x >= 1.0f)  { return 0.5f * FSW_PI_F; }
    if (x <= -1.0f) { return -0.5f * FSW_PI_F; }
    return fsw_atan2f(x, fsw_sqrtf(1.0f - x * x));
}

double fsw_sqrt_d(double x)
{
    double y;
    if (x != x) { return x; }
    if (x <= 0.0) { return 0.0; }
    if (x > 1.0e300) { return x; }
    if (x > 3.0e38 || x < 1.0e-37) {          /* outside float range: scale by an even power of two */
        double s = 1.0, r;
        while (x > 1.0e30) { x *= 1.0 / 1.099511627776e12; s *= 1.048576e6; }     /* 2^-40, 2^20 */
        while (x < 1.0e-30) { x *= 1.099511627776e12; s *= 1.0 / 1.048576e6; }
        r = fsw_sqrt_d(x);
        return r * s;
    }
    y = (double)fsw_sqrtf((float)x);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    return y;
}
