#include "fsw_watchdog.h"

void fsw_wdg_init(fsw_wdg_t *w, uint32_t required_mask, uint16_t miss_limit)
{
    w->required_mask = required_mask;
    w->checked_mask = 0;
    w->miss_count = 0;
    w->miss_limit = miss_limit;
    w->kicks = 0;
    w->tripped = 0;
}

void fsw_wdg_checkin(fsw_wdg_t *w, uint32_t task_bit)
{
    w->checked_mask |= task_bit;
}

int fsw_wdg_service(fsw_wdg_t *w)
{
    int ok = ((w->checked_mask & w->required_mask) == w->required_mask);
    w->checked_mask = 0;
    if (ok) {
        w->miss_count = 0;
        w->kicks++;
        return 1;
    }
    w->miss_count++;
    if (w->miss_count >= w->miss_limit) {
        w->tripped = 1;
    }
    return 0;
}
