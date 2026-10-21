/* Bounded, single-execution DMAT capture. The event probe and record layout
 * originate in the private Neutrino/ERETrace artifact (core_events.py).
 * Collection deliberately uses a synchronous path, separate from warp codecs.
 */
typedef struct hqsb_memory_source {
    CUfunction function; char *directory; struct hqsb_memory_source *next;
} hqsb_memory_source;
static hqsb_memory_source *hqsb_memory_sources;
static void hqsb_memory_register(CUfunction function,const char *directory) {
    hqsb_memory_source *item=malloc(sizeof(*item));
    if(!item)async_fatal("DMAT source allocation");
    item->function=function;item->directory=strdup(directory);
    item->next=hqsb_memory_sources;hqsb_memory_sources=item;
}
static const char *hqsb_memory_directory(CUfunction function) {
    for(hqsb_memory_source *p=hqsb_memory_sources;p;p=p->next)
        if(p->function==function)return p->directory;
    return "";
}
static CUresult hqsb_memory_launch(CUfunction probed, int nparam, void **params,
    unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void **extra) {
    size_t threads=(size_t)gx*gy*gz*bx*by*bz;
    size_t cap=strtoull(getenv("HQSB_DMAT_CAPACITY"),NULL,10);
    if (!cap || cap>2048 || threads>(256ULL<<20)/(cap*48+8) || !params || extra)
        async_fatal("DMAT capture exceeds 256 MiB or requires kernelParams launch");
    size_t bytes=threads*cap*48, counts_bytes=threads*8;
    CUdeviceptr records, counts;
    CUDA_CHECK(real_cuMemAlloc_v2(&records,bytes));
    CUDA_CHECK(real_cuMemAlloc_v2(&counts,counts_bytes));
    CUDA_CHECK(real_cuMemsetD32Async(records,0,bytes/4,stream));
    CUDA_CHECK(real_cuMemsetD32Async(counts,0,counts_bytes/4,stream));
    void **args=malloc((nparam+2)*sizeof(void*));
    void *host=malloc(bytes), *host_counts=malloc(counts_bytes);
    if(!args || !host || !host_counts)async_fatal("DMAT host allocation");
    memcpy(args,params,nparam*sizeof(void*));args[nparam]=&records;args[nparam+1]=&counts;
    if(shared)CUDA_CHECK(real_cuFuncSetAttribute(probed,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));
    CUresult rc=real_cuLaunchKernel(probed,gx,gy,gz,bx,by,bz,shared,stream,args,NULL);
    CUDA_CHECK(rc);
    /* Synchronize the submitting stream before reading its private buffers. */
    CUDA_CHECK(real_cuStreamSynchronize(stream));
    CUDA_CHECK(real_cuMemcpyDtoH_v2(host_counts,counts,counts_bytes));
    CUDA_CHECK(real_cuMemcpyDtoH_v2(host,records,bytes));
    char path[4096];
    snprintf(path,sizeof(path),"%s/memory-%llu.dmat",RESULT_DIR,hqsb_captured);
    FILE *out=fopen(path,"wb");
    uint32_t header[]={gx,gy,gz,bx,by,bz,shared,(uint32_t)cap};
    if(!out || fwrite("HQDMAT2\n",8,1,out)!=1 || fwrite(header,sizeof(header),1,out)!=1 ||
       fwrite(host_counts,counts_bytes,1,out)!=1 || fwrite(host,bytes,1,out)!=1)
        async_fatal("DMAT trace write");
    if(fclose(out))async_fatal("DMAT trace close");
    CUDA_CHECK(real_cuMemFree_v2(counts));CUDA_CHECK(real_cuMemFree_v2(records));
    free(args);free(host);free(host_counts);
    return rc;
}
