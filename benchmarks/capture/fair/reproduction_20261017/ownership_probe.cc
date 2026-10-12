// Finite ABI admission, not a performance generator. Reuse caller buffers at return.
#include <algorithm>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sched.h>
#include <stdexcept>
#include <string>

static void pin(int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu,&set);
    if (sched_setaffinity(0,sizeof(set),&set))throw std::runtime_error("affinity");
}
template<class T> static T read(std::ifstream& stream) {
    T value{};stream.read(reinterpret_cast<char*>(&value),sizeof(value));
    if(!stream)throw std::runtime_error("truncated input");
    return value;
}
int main(int argc,char**argv) {
    try {
        if(argc!=5)throw std::runtime_error("library encoded-path input-bin output-directory");
        pin(193);
        std::string library=argv[1];
        void* handle=dlopen(argv[1],RTLD_NOW|(library.find("python_bridge")!=std::string::npos?RTLD_GLOBAL:RTLD_LOCAL));
        if(!handle)throw std::runtime_error(dlerror());
        bool hindsight=library.find("hindsight")!=std::string::npos;
        auto open=reinterpret_cast<int(*)(const char*)>(dlsym(handle,hindsight?"hh_open":"hf_open"));
        auto offer=reinterpret_cast<int(*)(const char*,uint32_t,uint64_t,uint64_t,const char*)>(dlsym(handle,hindsight?"hh_offer":"hf_offer"));
        auto close=reinterpret_cast<int(*)()>(dlsym(handle,hindsight?"hh_close":"hf_close"));
        if(!open||!offer||!close||open(argv[2]))throw std::runtime_error("adapter open");
        pin(192);
        std::ifstream input(argv[3],std::ios::binary);
        uint32_t count=read<uint32_t>(input);
        for(uint32_t k=0;k<count;++k) {
            uint32_t p=read<uint32_t>(input);
            uint64_t seq=read<uint64_t>(input),ts=read<uint64_t>(input);
            uint32_t bytes=read<uint32_t>(input);
            if(bytes>65536)throw std::runtime_error("admission record bound");
            std::string body(bytes,'\0'),run="run_probe";
            input.read(body.data(),bytes);
            int status=input?offer(run.c_str(),p,seq,ts,body.c_str()):-1;
            if(status!=0&&status!=1)throw std::runtime_error("adapter offer");
            std::fill(body.begin(),body.end(),'!');
            std::fill(run.begin(),run.end(),'!');
        }
        if(close())throw std::runtime_error("adapter close");
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
