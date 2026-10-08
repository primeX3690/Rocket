#include "fsw_frames.h"
#include <math.h>

static void cross(const double a[3], const double b[3], double o[3])
{
    o[0] = a[1]*b[2] - a[2]*b[1];
    o[1] = a[2]*b[0] - a[0]*b[2];
    o[2] = a[0]*b[1] - a[1]*b[0];
}

void fsw_site_init(fsw_site_t *s, double lat, double lon, double r_pad, double az, double omega_e)
{
    double sl = sin(lat), cl = cos(lat), so = sin(lon), co = cos(lon);
    double E[3], N[3], U[3];
    int i;
    s->lat = lat; s->lon = lon; s->r_pad = r_pad; s->az = az; s->omega_e = omega_e;
    E[0] = -so;      E[1] = co;       E[2] = 0.0;
    N[0] = -sl * co; N[1] = -sl * so; N[2] = cl;
    U[0] = cl * co;  U[1] = cl * so;  U[2] = sl;
    for (i = 0; i < 3; i++) {
        s->bx[i] = sin(az) * E[i] + cos(az) * N[i];   /* downrange */
        s->bz[i] = U[i];
    }
    cross(s->bz, s->bx, s->by);                       /* Y = Z x X */
    /* spin axis (ECEF z) in I */
    s->axis[0] = s->bx[2];
    s->axis[1] = s->by[2];
    s->axis[2] = s->bz[2];
}

void fsw_site_pad_state(const fsw_site_t *s, double p[3], double v[3])
{
    double w[3], pp[3];
    pp[0] = 0.0; pp[1] = 0.0; pp[2] = s->r_pad;
    p[0] = pp[0]; p[1] = pp[1]; p[2] = pp[2];
    fsw_site_earth_rate(s, w);
    cross(w, pp, v);
}

void fsw_site_earth_rate(const fsw_site_t *s, double w[3])
{
    w[0] = s->omega_e * s->axis[0];
    w[1] = s->omega_e * s->axis[1];
    w[2] = s->omega_e * s->axis[2];
}

/* Rodrigues: rotate u about unit axis a by angle th. */
static void rot(const double a[3], double th, const double u[3], double o[3])
{
    double c = cos(th), sn = sin(th), d = a[0]*u[0] + a[1]*u[1] + a[2]*u[2], k[3];
    cross(a, u, k);
    o[0] = u[0]*c + k[0]*sn + a[0]*d*(1.0 - c);
    o[1] = u[1]*c + k[1]*sn + a[1]*d*(1.0 - c);
    o[2] = u[2]*c + k[2]*sn + a[2]*d*(1.0 - c);
}

void fsw_ecef_to_inertial(const fsw_site_t *s, const double ecef[3], double t, double out[3])
{
    double u[3];
    u[0] = s->bx[0]*ecef[0] + s->bx[1]*ecef[1] + s->bx[2]*ecef[2];
    u[1] = s->by[0]*ecef[0] + s->by[1]*ecef[1] + s->by[2]*ecef[2];
    u[2] = s->bz[0]*ecef[0] + s->bz[1]*ecef[1] + s->bz[2]*ecef[2];
    rot(s->axis, s->omega_e * t, u, out);                /* the Earth-fixed vector has turned by +omega t */
}

void fsw_inertial_to_ecef(const fsw_site_t *s, const double in[3], double t, double ecef[3])
{
    double u[3];
    rot(s->axis, -s->omega_e * t, in, u);
    ecef[0] = s->bx[0]*u[0] + s->by[0]*u[1] + s->bz[0]*u[2];
    ecef[1] = s->bx[1]*u[0] + s->by[1]*u[1] + s->bz[1]*u[2];
    ecef[2] = s->bx[2]*u[0] + s->by[2]*u[1] + s->bz[2]*u[2];
}

void fsw_polar_from_inertial(const double p[3], const double v[3], double *r, double *speed, double *gamma, double *psi)
{
    double rr = sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    double vv = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    double vr = (rr > 0.0) ? (p[0]*v[0] + p[1]*v[1] + p[2]*v[2]) / rr : 0.0;
    double s = (vv > 0.0) ? vr / vv : 0.0;
    if (s > 1.0) { s = 1.0; }
    if (s < -1.0) { s = -1.0; }
    *r = rr; *speed = vv; *gamma = asin(s); *psi = atan2(p[0], p[2]);
}
