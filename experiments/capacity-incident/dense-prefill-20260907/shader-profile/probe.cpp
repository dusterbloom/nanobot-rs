#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "mlx/mlx.h"
#include "mlx/backend/metal/metal.h"
using namespace mlx::core;
int main(int argc,char**argv){try{
 set_default_device(Device::gpu);set_cache_limit(256*1024*1024);
 int nq=std::atoi(argv[1]),nk=std::atoi(argv[2]);random::seed(20260908);
 // Match the serving Q transpose and padded KV backing, with materialized inputs.
 auto q=transpose(random::normal({1,nq,16,256}),{0,2,1,3});
 auto k=slice(random::normal({1,2,nk+256,256}),Shape{0,0,0,0},Shape{1,2,nk,256});
 auto v=slice(random::normal({1,2,nk+256,256}),Shape{0,0,0,0},Shape{1,2,nk,256});
 auto mask=greater_equal(reshape(arange(nk-nq,nk),{nq,1}),reshape(arange(nk),{1,nk}));eval(q,k,v,mask);
 std::vector<std::string> names={"fallback","query128","original","plain","reg","original_causal","plain_causal","reg_causal"};
 auto run=[&](int arm,bool causal=false){
  int kernel_arm=arm>=5?arm-3:arm;
  if(arm>=5)causal=true;
  if(arm>=2)setenv("HIGGS_PROBE_D256",names[kernel_arm].c_str(),1);else unsetenv("HIGGS_PROBE_D256");
  if(arm!=1)return fast::scaled_dot_product_attention(q,k,v,0.0625f,causal?"causal":"array",causal?std::nullopt:std::optional<array>(mask));
  std::vector<array> out;
  for(int start=0;start<nq;start+=128){int end=std::min(start+128,nq);
   auto qs=slice(q,Shape{0,0,start,0},Shape{1,16,end,256});auto ms=slice(mask,Shape{start,0},Shape{end,nk});
   auto o=fast::scaled_dot_product_attention(qs,k,v,0.0625f,"array",ms);eval(o);out.push_back(o);
  }return concatenate(out,2);
 };
 int arm=std::atoi(argv[3]);
 for(int warm=0;warm<3;warm++){auto out=run(arm);eval(out);}
 synchronize();
 metal::start_capture(argv[4]);
 auto out=run(arm);eval(out);synchronize();
 metal::stop_capture();
 std::cout<<"CAPTURE COMPLETE "<<names[arm]<<std::endl;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
