#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER hqsb_capture
#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "lttng_provider.h"
#if !defined(HQSB_LTTNG_PROVIDER_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define HQSB_LTTNG_PROVIDER_H
#include <stdint.h>
#include <lttng/tracepoint.h>
LTTNG_UST_TRACEPOINT_EVENT(hqsb_capture, event,
    LTTNG_UST_TP_ARGS(const char*, run_id, uint32_t, producer_id, uint64_t, source_seq,
                     uint64_t, source_timestamp, const char*, body),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_string(run_id, run_id)
                        lttng_ust_field_integer(uint32_t, producer_id, producer_id)
                        lttng_ust_field_integer(uint64_t, source_seq, source_seq)
                        lttng_ust_field_integer(uint64_t, source_timestamp, source_timestamp)
                        lttng_ust_field_string(body, body)))
#endif
#include <lttng/tracepoint-event.h>
