/* fsw_frames.h -- reference frames for orbit-scale navigation.
 *
 *  I  : the navigation (inertial) frame. NON-ROTATING, origin at the Earth's centre, axes fixed at the
 *       moment navigation starts (liftoff, t_L):
 *         +Z = local vertical at the pad ("up"),
 *         +X = launch (downrange) direction, horizontal, azimuth `az` from north,
 *         +Y = Z x X (left of downrange).
 *       The pad therefore starts at (0, 0, r_pad). The vehicle body frame is aligned with it on the pad
 *       (+Z axial/up, +X downrange, +Y = pitch axis), which is exactly the convention of the flat-Earth
 *       filter, so pitch about +Y and the PEG plane are the same objects in both.
 *  E  : ECEF, what a GNSS receiver reports (rotating with the Earth).
 *
 * Spherical Earth (geocentric = geodetic latitude; r_pad is the geocentric radius of the pad). The
 * ellipsoid is a ~21 km radial / 0.2 deg direction effect at mid-latitudes that must be added before
 * flying (see README). Earth rotation rate defaults to 7.2921150e-5 rad/s.
 *
 * Uses libm in double (like fsw_peg.c): this is the only other libm user. */
#ifndef FSW_FRAMES_H
#define FSW_FRAMES_H

#define FSW_OMEGA_EARTH 7.2921150e-5

typedef struct {
    double lat, lon, r_pad, az, omega_e;
    double bx[3], by[3], bz[3];   /* I-frame axes expressed in ECEF (rows of the ECEF->I matrix)  */
    double axis[3];               /* Earth spin axis, unit, expressed in I                         */
} fsw_site_t;

void fsw_site_init(fsw_site_t *s, double lat_rad, double lon_rad, double r_pad_m, double azimuth_rad, double omega_e);

/* Inertial position/velocity of the pad at t_L: p = (0,0,r_pad), v = omega x p. */
void fsw_site_pad_state(const fsw_site_t *s, double p[3], double v[3]);

/* Earth spin vector omega_e * axis, expressed in I. */
void fsw_site_earth_rate(const fsw_site_t *s, double w[3]);

/* ECEF position -> I, for a receiver fix valid `t_since_L_s` after t_L (and the inverse). */
void fsw_ecef_to_inertial(const fsw_site_t *s, const double ecef[3], double t_since_L_s, double out[3]);
void fsw_inertial_to_ecef(const fsw_site_t *s, const double in[3], double t_since_L_s, double ecef[3]);

/* Polar-frame quantities PEG needs. r: radius; speed: INERTIAL speed |v|; gamma: inertial flight-path
 * angle from the local horizontal; psi: angle of the position vector from the pad vertical, in the X-Z
 * plane (what rotates the local horizontal away from the pad horizontal). */
void fsw_polar_from_inertial(const double p[3], const double v[3], double *r, double *speed, double *gamma, double *psi);

#endif
