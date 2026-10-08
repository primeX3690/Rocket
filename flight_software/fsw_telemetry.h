/* fsw_telemetry.h -- framed, CRC-protected telemetry link.
 *
 * Frame:  EB 90 | type | seq | len | payload[len] | crc16_hi | crc16_lo
 * CRC-16/CCITT-FALSE covers type,seq,len,payload (not the sync bytes).
 *
 * The parser is a byte-at-a-time resynchronising state machine: after a
 * corrupted frame it drops ONE byte and rescans what it already buffered,
 * so a valid frame that started inside a corrupted one is not lost.
 * No malloc, fixed buffers, safe to run from a UART RX interrupt. */
#ifndef FSW_TELEMETRY_H
#define FSW_TELEMETRY_H
#include <stddef.h>
#include <stdint.h>

#define FSW_TLM_SYNC1        0xEBu
#define FSW_TLM_SYNC2        0x90u
#define FSW_TLM_MAX_PAYLOAD  64u
#define FSW_TLM_HEADER       5u
#define FSW_TLM_MAX_FRAME    (FSW_TLM_HEADER + FSW_TLM_MAX_PAYLOAD + 2u)

typedef struct {
    uint8_t type;
    uint8_t seq;
    uint8_t len;
    uint8_t payload[FSW_TLM_MAX_PAYLOAD];
} fsw_tlm_frame_t;

/* Returns frame size written, or 0 if `cap` is too small / len too big. */
size_t fsw_tlm_encode(uint8_t *out, size_t cap, uint8_t type, uint8_t seq,
                      const uint8_t *payload, uint8_t len);

typedef struct {
    uint8_t  buf[FSW_TLM_MAX_FRAME];
    size_t   n;
    uint32_t frames_ok;
    uint32_t crc_errors;
    uint32_t bytes_dropped;
} fsw_tlm_parser_t;

void fsw_tlm_parser_init(fsw_tlm_parser_t *p);
/* Feed one byte. Returns 1 if a valid frame is now available in *out.
 * After a 1, call fsw_tlm_parser_poll() until it returns 0 to drain any
 * further complete frames that were already buffered during a resync. */
int  fsw_tlm_parser_feed(fsw_tlm_parser_t *p, uint8_t byte, fsw_tlm_frame_t *out);
int  fsw_tlm_parser_poll(fsw_tlm_parser_t *p, fsw_tlm_frame_t *out);

#endif
