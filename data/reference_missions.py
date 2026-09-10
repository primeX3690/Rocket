"""
data/reference_missions.py

Publicly published, order-of-magnitude reference parameters for a
PSLV-class 4-stage launch vehicle, used ONLY to sanity-check that this
project's multi-stage mission simulation produces physically plausible
results (burn durations, altitudes, orbital insertion) — NOT an attempt
to reproduce exact proprietary mission telemetry, which is neither
available nor needed for that purpose.

Sources: figures of this general order are widely published in ISRO
mission press-kits, public program overviews, and open aerospace
textbooks/course material describing PSLV-class vehicles. Treat all
values here as APPROXIMATE, PUBLIC, NON-SENSITIVE order-of-magnitude
reference points for validation — not as precise or authoritative
mission data.
"""

# Approximate, publicly-known order-of-magnitude parameters for a
# PSLV-class mission targeting a ~500-620 km sun-synchronous polar orbit
# (a common PSLV mission class). All values are representative rather
# than mission-specific.

PSLV_CLASS_REFERENCE = {
    "typical_target_altitude_km": (500, 620),          # SSO-class missions
    "typical_stage1_burn_duration_s": (100, 115),        # core (solid) stage
    "typical_stage1_burnout_altitude_km": (55, 70),
    "typical_total_ascent_to_orbit_duration_s": (900, 1100),  # ~15-18 min, widely reported
    "typical_total_mission_dv_m_s": (9200, 9700),       # standard LEO delta-V budget class
    "typical_payload_to_500km_sso_kg": (1200, 1750),     # representative payload class
}


def sanity_check_mission_result(result: dict) -> dict:
    """
    Compare a simulated multi-stage mission result against the
    PSLV_CLASS_REFERENCE order-of-magnitude ranges above and report
    which checks pass. This is a PLAUSIBILITY check, not a precision
    validation — it answers "is this simulation in the right ballpark
    of a real launch-vehicle-class mission, or has something gone
    wrong by orders of magnitude?"
    """
    ref = PSLV_CLASS_REFERENCE
    checks = {}

    total_time = result.get("total_mission_time_s")
    if total_time is not None:
        lo, hi = ref["typical_total_ascent_to_orbit_duration_s"]
        checks["total_ascent_duration_plausible"] = lo * 0.3 <= total_time <= hi * 1.5
        # wide tolerance band: this project's simplified point-mass
        # vehicle is not a mass/thrust match for any specific real
        # vehicle, so we check order-of-magnitude plausibility only

    final_alt_km = result.get("final_altitude_km")
    if final_alt_km is not None:
        lo, hi = ref["typical_target_altitude_km"]
        checks["final_altitude_in_plausible_leo_sso_range"] = 150.0 <= final_alt_km <= hi * 1.5

    return checks
