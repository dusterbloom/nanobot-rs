#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "mlx/mlx.h"
using namespace mlx::core;
int main(int argc, char** argv) {
  try {
    set_default_device(Device::gpu); set_cache_limit(256*1024*1024);
    int nq=argc>1?std::atoi(argv[1]):32, nk=argc>2?std::atoi(argv[2]):1024;
    random::seed(20260907);
    auto q=random::normal({1,16,nq,256}); auto k=random::normal({1,2,nk,256}); auto v=random::normal({1,2,nk,256});
    auto mask=greater_equal(reshape(arange(nk-nq,nk),{nq,1}),reshape(arange(nk),{1,nk}));
    eval(q,k,v,mask);
    auto run=[&](bool candidate, bool causal){
      if(candidate) setenv("HIGGS_PROBE_D256","1",1); else unsetenv("HIGGS_PROBE_D256");
      return fast::scaled_dot_product_attention(q,k,v,0.0625f,causal?"causal":"array",causal?std::nullopt:std::optional<array>(mask));
    };
    auto ref=run(false,false); eval(ref);
    auto tiled=[&](int tile){
      unsetenv("HIGGS_PROBE_D256");
      std::vector<array> rows;
      for(int start=0;start<nq;start+=tile){
        int end=std::min(start+tile,nq),kend=nk-nq+end;
        auto qs=slice(q,Shape{0,0,start,0},Shape{1,16,end,256});
        auto ks=slice(k,Shape{0,0,0,0},Shape{1,2,kend,256});
        auto vs=slice(v,Shape{0,0,0,0},Shape{1,2,kend,256});
        auto out=fast::scaled_dot_product_attention(qs,ks,vs,0.0625f,"causal");
        eval(out); rows.push_back(out);
      }
      return concatenate(rows,2);
    };
    for(int tile:{64,128,256,512}) {
      auto candidate=tiled(tile); eval(candidate);
      auto error=max(abs(subtract(ref,candidate)));
      auto rel=sqrt(divide(sum(square(subtract(ref,candidate))),sum(square(ref)))); eval(error,rel);
      std::cout<<"{\"tile\":"<<tile<<",\"max_abs\":"<<error.item<float>()<<",\"relative_l2\":"<<rel.item<float>()<<"}"<<std::endl;
      if(!std::isfinite(error.item<float>()) || error.item<float>()>1e-5f || rel.item<float>()>1e-5f) return 2;
    }
    for(int tile:{1024,64,128,256,512,512,256,128,64,1024}) {
      std::vector<double> times;
      for(int rep=0;rep<4;rep++) { auto t=std::chrono::steady_clock::now(); auto out=tiled(tile); eval(out); auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count(); if(rep)times.push_back(ms); }
      std::sort(times.begin(),times.end());std::cout<<"{\"nq\":"<<nq<<",\"nk\":"<<nk<<",\"tile\":"<<tile<<",\"median_ms\":"<<times[1]<<"}"<<std::endl;
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
