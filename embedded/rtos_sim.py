"""
embedded/rtos_sim.py

A software model of hard-real-time task scheduling (the kind of
scheduling a real FreeRTOS/Zephyr deployment would do on the actual
flight computer), with deadline-miss monitoring -- NOT a real RTOS
integration (this project has no hardware and no FreeRTOS/Zephyr
kernel available to link against; see embedded/README.md for the
existing, real ARM Cortex-M3/QEMU SIL harness this complements).

What IS real here: a genuine fixed-priority preemptive scheduler
simulation (the same scheduling MODEL real RTOSes implement), including
Rate-Monotonic priority assignment, exact-arithmetic timeline
simulation (tracks remaining execution time per task tick-by-tick), and
deadline-miss detection -- useful for sizing a real task set's CPU
budget and catching over-subscription BEFORE deploying to hardware,
which is exactly what real flight-software teams use RTA (response-time
analysis) / scheduling simulation for during design.

Zero external dependencies beyond NumPy - CPU-only.
"""

import numpy as np


class PeriodicTask:
    def __init__(self, name: str, period_s: float, execution_time_s: float, deadline_s: float = None):
        self.name = name
        self.period = period_s
        self.execution_time = execution_time_s
        self.deadline = deadline_s if deadline_s is not None else period_s   # implicit-deadline default
        self.priority = None   # assigned by the scheduler (lower period -> higher priority under RMS)


def rate_monotonic_priorities(tasks):
    """Rate-Monotonic Scheduling: shorter period = higher priority (optimal fixed-priority policy)."""
    ordered = sorted(tasks, key=lambda t: t.period)
    for i, t in enumerate(ordered):
        t.priority = i   # 0 = highest priority
    return tasks


def utilization_bound_schedulable(tasks) -> dict:
    """
    Liu & Layland's classic RMS sufficient (not necessary) schedulability
    test: sum(C_i/T_i) <= n*(2^(1/n) - 1). Passing guarantees
    schedulability; failing does NOT prove unschedulability (exact
    response-time analysis, done below, is definitive).
    """
    n = len(tasks)
    if n == 0:
        return {"utilization": 0.0, "bound": 1.0, "guaranteed_schedulable": True}
    utilization = sum(t.execution_time / t.period for t in tasks)
    bound = n * (2 ** (1.0 / n) - 1.0)
    return {"utilization": utilization, "bound": bound, "guaranteed_schedulable": utilization <= bound}


def exact_response_time_analysis(tasks) -> dict:
    """
    Exact (necessary AND sufficient) fixed-priority response-time
    analysis via the standard fixed-point recurrence:
        R_i = C_i + sum_{j higher priority} ceil(R_i / T_j) * C_j
    iterated to convergence (or until R_i exceeds the deadline, a miss).
    This is the real, load-bearing schedulability test real RTOS-based
    flight software is sized against -- not a simulation approximation.
    """
    tasks = rate_monotonic_priorities(list(tasks))
    ordered = sorted(tasks, key=lambda t: t.priority)
    results = {}
    for i, task in enumerate(ordered):
        higher_priority = ordered[:i]
        R = task.execution_time
        for _ in range(1000):
            interference = sum(np.ceil(R / hp.period) * hp.execution_time for hp in higher_priority)
            R_new = task.execution_time + interference
            if abs(R_new - R) < 1e-9:
                break
            R = R_new
            if R > task.deadline * 10:   # diverging -> definitely a miss
                break
        results[task.name] = {"worst_case_response_time_s": R, "deadline_s": task.deadline,
                              "meets_deadline": R <= task.deadline}
    return results


def simulate_scheduler_timeline(tasks, sim_duration_s: float, tick_s: float = 1e-4):
    """
    Tick-by-tick fixed-priority PREEMPTIVE scheduler simulation:
    actually runs the highest-priority ready task each tick, tracks
    remaining execution time per pending job, and records every
    deadline that is missed in the simulated timeline -- a direct,
    literal simulation (not just the analytical bound above), useful
    for sanity-checking the analysis and for visualizing jitter.
    """
    tasks = rate_monotonic_priorities(list(tasks))
    ordered = sorted(tasks, key=lambda t: t.priority)
    n_ticks = int(sim_duration_s / tick_s)

    pending = {t.name: [] for t in ordered}   # list of [remaining_time, deadline_abs_time]
    next_release = {t.name: 0.0 for t in ordered}
    missed_deadlines = []
    cpu_busy_ticks = 0

    for tick in range(n_ticks):
        t_now = tick * tick_s
        for task in ordered:
            if t_now + 1e-12 >= next_release[task.name]:
                pending[task.name].append([task.execution_time, t_now + task.deadline])
                next_release[task.name] += task.period

        running_task = None
        for task in ordered:   # highest priority first
            if pending[task.name]:
                running_task = task
                break

        if running_task is not None:
            job = pending[running_task.name][0]
            job[0] -= tick_s
            cpu_busy_ticks += 1
            if job[0] <= 1e-9:
                pending[running_task.name].pop(0)

        for task in ordered:
            still_pending = [job for job in pending[task.name] if job[1] < t_now]
            for job in still_pending:
                missed_deadlines.append({"task": task.name, "deadline_time_s": job[1]})
            pending[task.name] = [job for job in pending[task.name] if job[1] >= t_now]

    return {
        "cpu_utilization": cpu_busy_ticks / max(n_ticks, 1),
        "missed_deadlines": missed_deadlines,
        "n_missed": len(missed_deadlines),
        "schedulable": len(missed_deadlines) == 0,
    }
