/* Selective Triton loader bridge. CUDA runtime/cuBLAS keep the real driver.
 * The ERE hook is local and deep-bound so its internal calls keep their ABI.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
static void *(*next_open)(const char*,int);
static void *driver, *hook;
static pthread_once_t loader_once=PTHREAD_ONCE_INIT;
static void loader(void){
    next_open=dlsym(RTLD_NEXT,"dlopen");
}
static pthread_once_t driver_once=PTHREAD_ONCE_INIT;
static void load_driver(void){
    pthread_once(&loader_once,loader);
    driver=next_open(getenv("NEUTRINO_REAL_DRIVER"),RTLD_NOW|RTLD_LOCAL);
    if(!driver){fprintf(stderr,"HQSB ERE real driver: %s\n",dlerror());abort();}
}
static pthread_mutex_t hook_mutex=PTHREAD_MUTEX_INITIALIZER;
static void *hook_symbol(const char *name){
    pthread_once(&loader_once,loader);
    pthread_mutex_lock(&hook_mutex);
    if(!hook)hook=next_open(getenv("NEUTRINO_HOOK_DRIVER"),RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);
    void *result=hook?dlsym(hook,name):NULL;
    pthread_mutex_unlock(&hook_mutex);
    if(!result){fprintf(stderr,"HQSB ERE hook symbol %s: %s\n",name,dlerror());abort();}
    return result;
}
static int triton_caller(void *address){
    Dl_info info;
    return dladdr(address,&info)&&info.dli_fname&&
        (strstr(info.dli_fname,"cuda_utils")||strstr(info.dli_fname,"__triton_launcher"));
}
static void *target(const char *name,void *caller){
    pthread_once(&driver_once,load_driver);
    return triton_caller(caller)?hook_symbol(name):dlsym(driver,name);
}
void *dlopen(const char *path,int flags){
    pthread_once(&loader_once,loader);
    if(path&&strstr(path,"libcuda.so")&&triton_caller(__builtin_return_address(0))){
        hook_symbol("cuInit");return hook;
    }
    return next_open(path,flags);
}
int cuModuleLoadDataEx(void **module,const void *image,unsigned n,int *options,void **values){
    int(*f)(void**,const void*,unsigned,int*,void**)=target("cuModuleLoadDataEx",__builtin_return_address(0));
    return f(module,image,n,options,values);
}
int cuModuleLoadData(void **module,const void *image){
    int(*f)(void**,const void*)=target("cuModuleLoadData",__builtin_return_address(0));return f(module,image);
}
int cuModuleGetFunction(void **function,void *module,const char *name){
    int(*f)(void**,void*,const char*)=target("cuModuleGetFunction",__builtin_return_address(0));return f(function,module,name);
}
int cuLaunchKernel(void *function,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,void *stream,void **args,void **extra){
    int(*f)(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**)=target("cuLaunchKernel",__builtin_return_address(0));
    return f(function,gx,gy,gz,bx,by,bz,shared,stream,args,extra);
}
int cuLaunchKernelEx(const void *config,void *function,void **args,void **extra){
    int(*f)(const void*,void*,void**,void**)=target("cuLaunchKernelEx",__builtin_return_address(0));return f(config,function,args,extra);
}
/* Runtime profiler controls select the ERE window; other API pointers remain
 * exactly those returned by the installed driver. v1 has four arguments. */
int cuGetProcAddress(const char *name,void **pointer,int version,unsigned long long flags){
    pthread_once(&driver_once,load_driver);
    int(*f)(const char*,void**,int,unsigned long long)=dlsym(driver,"cuGetProcAddress");
    int rc=f(name,pointer,version,flags);
    if(!rc&&(!strcmp(name,"cuProfilerStart")||!strcmp(name,"cuProfilerStop")))*pointer=hook_symbol(name);
    return rc;
}
int cuGetProcAddress_v2(const char *name,void **pointer,int version,unsigned long long flags,int *status){
    pthread_once(&driver_once,load_driver);
    int(*f)(const char*,void**,int,unsigned long long,int*)=dlsym(driver,"cuGetProcAddress_v2");
    int rc=f(name,pointer,version,flags,status);
    if(!rc&&(!strcmp(name,"cuProfilerStart")||!strcmp(name,"cuProfilerStop")))*pointer=hook_symbol(name);
    return rc;
}
