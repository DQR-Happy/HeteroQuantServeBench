/* HQSB adapter for the private ERETrace artifact. Included after ablation.h.
 * One CUDA context and one submitting thread own an explicitly opened window.
 * Captured launches keep the artifact's arena/recovery protocol unchanged.
 */
#include <stdatomic.h>
static void init(void);
static _Atomic int hqsb_active;
static _Atomic int hqsb_done;
static pthread_mutex_t hqsb_control_mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_t hqsb_producer;
static _Atomic unsigned long long hqsb_seen, hqsb_other_thread;
static unsigned long long hqsb_matching, hqsb_attempts, hqsb_captured;
static FILE *hqsb_launches;
static CUcontext hqsb_capture_context;
/* Producer-owned tensor descriptors are snapshotted into each launch record.
 * JSON is created by our child-only Python adapter, never by a browser. */
static _Thread_local char *hqsb_dmat_context;
int hqsb_eretrace_context(const char *json) {
    if(json && strlen(json)>65536)return -1;
    char *next=json?strdup(json):NULL;
    if(json && !next)return -1;
    free(hqsb_dmat_context);hqsb_dmat_context=next;return 0;
}
#include "hqsb_memory.h"
static CUresult hqsb_capture_launch(CUfunction, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);

int hqsb_eretrace_start(void) {
    if (shared_lib == NULL) init();
    pthread_mutex_lock(&hqsb_control_mutex);
    if (atomic_load(&hqsb_active) || hqsb_done) {
        pthread_mutex_unlock(&hqsb_control_mutex); return 0;
    }
    hqsb_producer = pthread_self();
    hqsb_seen = hqsb_other_thread = 0;
    hqsb_matching = hqsb_attempts = hqsb_captured = 0;
    hqsb_capture_context = NULL;
    if (getenv("HQSB_ERE_CODEC_RAW")) ar_codec_min=SIZE_MAX;
    char path[4096];
    snprintf(path, sizeof(path), "%s/hqsb-launches.jsonl", RESULT_DIR);
    hqsb_launches = fopen(path, "w");
    if (!hqsb_launches) { pthread_mutex_unlock(&hqsb_control_mutex); return -1; }
    atomic_store(&hqsb_active, 1);
    pthread_mutex_unlock(&hqsb_control_mutex);
    return 0;
}
int hqsb_eretrace_stop(void) {
    if (!atomic_load(&hqsb_active)) return 0;
    if (!pthread_equal(pthread_self(), hqsb_producer)) return -1;
    hqsb_done=1;
    atomic_store(&hqsb_active, 0);
    if (neutrino_ablation_boundary()) return -1;
    char path[4096];
    snprintf(path, sizeof(path), "%s/arena-stats.json", RESULT_DIR);
    if (neutrino_ablation_flush(path)) return -1;
    if (fclose(hqsb_launches)) return -1;
    hqsb_launches = NULL;
    snprintf(path, sizeof(path), "%s/coverage.json", RESULT_DIR);
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    fprintf(fp, "{\"routed_launches\":%llu,\"other_thread_launches\":%llu,"
        "\"matching_launches\":%llu,\"attempted_launches\":%llu,"
        "\"captured_launches\":%llu,\"probe_failed_launches\":%llu,"
        "\"scope\":\"selected_ordinary_launches_one_producer_one_context\"}\n",
        (unsigned long long)hqsb_seen, (unsigned long long)hqsb_other_thread,
        hqsb_matching, hqsb_attempts, hqsb_captured, hqsb_attempts-hqsb_captured);
    int bad=ferror(fp); if(fclose(fp))bad=1;
    return bad ? -1 : 0;
}
CUresult cuProfilerStart(void) {
    if (shared_lib == NULL) init();
    return (hqsb_done || atomic_load(&hqsb_active) || !hqsb_eretrace_start()) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
CUresult cuProfilerStop(void) {
    return hqsb_eretrace_stop() ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}
CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream,
    void **args, void **extra) {
    if (shared_lib == NULL) init();
    const char *window=getenv("HQSB_ERE_WINDOW_FILE");
    if(window && !hqsb_done && !atomic_load(&hqsb_active) && access(window,F_OK)==0)
        if(hqsb_eretrace_start())async_fatal("HQSB capture window start");
    if(window && atomic_load(&hqsb_active) && access(window,F_OK)!=0 && pthread_equal(pthread_self(),hqsb_producer))
        if(hqsb_eretrace_stop())async_fatal("HQSB capture window drain");
    if (!atomic_load(&hqsb_active) || ar_external_depth)
        return real_cuLaunchKernel(f,gx,gy,gz,bx,by,bz,shared,stream,args,extra);
    atomic_fetch_add(&hqsb_seen,1);
    if (!pthread_equal(pthread_self(),hqsb_producer)) {
        atomic_fetch_add(&hqsb_other_thread,1);
        return real_cuLaunchKernel(f,gx,gy,gz,bx,by,bz,shared,stream,args,extra);
    }
    char *name=NULL; size_t size; void *binary;
    const char *selector=getenv("HQSB_ERE_KERNEL");
    if (binmap_get(f,&size,&name,&binary)==-1 || !name || !selector || !strstr(name,selector))
        return real_cuLaunchKernel(f,gx,gy,gz,bx,by,bz,shared,stream,args,extra);
    unsigned long long index=hqsb_matching++;
    unsigned long long skip=strtoull(getenv("HQSB_ERE_SKIP"),NULL,10);
    unsigned long long count=strtoull(getenv("HQSB_ERE_COUNT"),NULL,10);
    if(index<skip || hqsb_attempts>=count)
        return real_cuLaunchKernel(f,gx,gy,gz,bx,by,bz,shared,stream,args,extra);
    size_t records=(size_t)gx*gy*gz*(((size_t)bx*by*bz+31)/32);
    if(records>(16ULL<<20)/16)async_fatal("HQSB selected Map exceeds 16 MiB capture bound");
    CUcontext context; CUDA_CHECK(real_cuCtxGetCurrent(&context));
    if(hqsb_capture_context && hqsb_capture_context!=context)async_fatal("HQSB capture window requires one CUDA context");
    hqsb_capture_context=context;
    if(ar_engines && ar_engines->context!=context)async_fatal("HQSB capture window requires one CUDA context");
    hqsb_attempts++;
    CUresult rc=hqsb_capture_launch(f,gx,gy,gz,bx,by,bz,shared,stream,args,extra);
    int nparam,nprobe,*sizes,*types; bool success=false; CUfunction probed,pruned,countd; char *compiled_name;
    if(funcmap_get(f,&compiled_name,&nparam,&nprobe,&sizes,&types,&success,(void**)&probed,(void**)&pruned,(void**)&countd)==-1)success=false;
    fprintf(hqsb_launches,"{\"attempt\":%llu,\"captured\":%s,\"frame\":%llu,\"kernel\":",
        hqsb_attempts-1,success&&rc==CUDA_SUCCESS?"true":"false",hqsb_captured);
    nt_json_string(hqsb_launches,name);
    fprintf(hqsb_launches,",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u]",gx,gy,gz,bx,by,bz);
    if(getenv("HQSB_DMAT_CAPACITY")) {
        fputs(",\"memory_metadata\":",hqsb_launches);
        nt_json_string(hqsb_launches,hqsb_memory_directory(f));
        fprintf(hqsb_launches,",\"tensor_context\":%s",hqsb_dmat_context?hqsb_dmat_context:"{}");
    }
    fputs("}\n",hqsb_launches);
    if(ferror(hqsb_launches))async_fatal("HQSB launch lineage write");
    /* Memory compilation registers two thread Maps; the arena warp reader
     * retains its existing single-Map protocol. */
    if(success&&rc==CUDA_SUCCESS)hqsb_captured++;
    /* A bounded selection closes on its producer, with the producer CUDA
     * context still current. This also works with upstream profiler routes
     * that resolve CUDA runtime controls directly from the native driver. */
    if(window && hqsb_attempts>=count)
        if(hqsb_eretrace_stop())async_fatal("HQSB capture window drain");
    return rc;
}
