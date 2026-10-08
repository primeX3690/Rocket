#include "fsw_telemetry.h"
#include "fsw_crc.h"
#include <string.h>

size_t fsw_tlm_encode(uint8_t *out, size_t cap, uint8_t type, uint8_t seq,
                      const uint8_t *payload, uint8_t len)
{
    size_t total = FSW_TLM_HEADER + (size_t)len + 2u;
    uint16_t crc;
    if (len > FSW_TLM_MAX_PAYLOAD || cap < total) {
        return 0;
    }
    out[0] = (uint8_t)FSW_TLM_SYNC1;
    out[1] = (uint8_t)FSW_TLM_SYNC2;
    out[2] = type;
    out[3] = seq;
    out[4] = len;
    if (len > 0u) {
        memcpy(&out[FSW_TLM_HEADER], payload, len);
    }
    crc = fsw_crc16_ccitt(&out[2], 3u + (size_t)len, 0xFFFFu);
    out[FSW_TLM_HEADER + len]      = (uint8_t)(crc >> 8);
    out[FSW_TLM_HEADER + len + 1u] = (uint8_t)(crc & 0xFFu);
    return total;
}

void fsw_tlm_parser_init(fsw_tlm_parser_t *p)
{
    memset(p, 0, sizeof(*p));
}

static void discard(fsw_tlm_parser_t *p, size_t k)
{
    memmove(p->buf, p->buf + k, p->n - k);
    p->n -= k;
    p->bytes_dropped += (uint32_t)k;
}

int fsw_tlm_parser_poll(fsw_tlm_parser_t *p, fsw_tlm_frame_t *out)
{
    for (;;) {
        size_t i = 0;
        size_t len, total, crc_len;
        uint16_t crc, rx;

        while (i < p->n && p->buf[i] != FSW_TLM_SYNC1) {
            i++;
        }
        if (i > 0u) {
            discard(p, i);
        }
        if (p->n == 0u) {
            return 0;
        }
        if (p->n >= 2u && p->buf[1] != FSW_TLM_SYNC2) {
            discard(p, 1u);
            continue;
        }
        if (p->n < FSW_TLM_HEADER) {
            return 0;
        }
        len = p->buf[4];
        if (len > FSW_TLM_MAX_PAYLOAD) {
            discard(p, 1u);
            continue;
        }
        total = FSW_TLM_HEADER + len + 2u;
        if (p->n < total) {
            return 0;
        }
        crc_len = 3u + len;
        crc = fsw_crc16_ccitt(&p->buf[2], crc_len, 0xFFFFu);
        rx  = (uint16_t)(((uint16_t)p->buf[FSW_TLM_HEADER + len] << 8) |
                          (uint16_t)p->buf[FSW_TLM_HEADER + len + 1u]);
        if (crc == rx) {
            out->type = p->buf[2];
            out->seq  = p->buf[3];
            out->len  = (uint8_t)len;
            memcpy(out->payload, &p->buf[FSW_TLM_HEADER], len);
            memmove(p->buf, p->buf + total, p->n - total);
            p->n -= total;
            p->frames_ok++;
            return 1;
        }
        p->crc_errors++;
        discard(p, 1u);
    }
}

int fsw_tlm_parser_feed(fsw_tlm_parser_t *p, uint8_t byte, fsw_tlm_frame_t *out)
{
    /* Buffer can never overflow: poll() only returns 0 while n < total <= MAX_FRAME. */
    p->buf[p->n++] = byte;
    return fsw_tlm_parser_poll(p, out);
}
