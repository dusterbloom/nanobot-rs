#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "mlx/mlx.h"
using namespace mlx::core;
int main(int argc,char**argv){try{
 set_default_device(Device::gpu);set_cache_limit(256*1024*1024);
 int nq=std::atoi(argv[1]),nk=std::atoi(argv[2]);random::seed(20260908);
 // Match the serving Q transpose and padded KV backing, with materialized inputs.
 auto q=transpose(random::normal({1,nq,16,256}),{0,2,1,3});
 auto k=slice(random::normal({1,2,nk+256,256}),Shape{0,0,0,0},Shape{1,2,nk,256});
 auto v=slice(random::normal({1,2,nk+256,256}),Shape{0,0,0,0},Shape{1,2,nk,256});
 auto mask=greater_equal(reshape(arange(nk-nq,nk),{nq,1}),reshape(arange(nk),{1,nk}));eval(q,k,v,mask);
 std::vector<std::string> names={"fallback","query128","reg","reg_causal","peel_causal"};
 auto run=[&](int arm,bool causal=false){
  if(arm>=3)causal=true;
  if(arm>=2)setenv("HIGGS_PROBE_D256",arm==4?"peel":"reg",1);else unsetenv("HIGGS_PROBE_D256");
  if(arm!=1)return fast::scaled_dot_product_attention(q,k,v,0.0625f,causal?"causal":"array",causal?std::nullopt:std::optional<array>(mask));
  std::vector<array> out;
  for(int start=0;start<nq;start+=128){int end=std::min(start+128,nq);
   auto qs=slice(q,Shape{0,0,start,0},Shape{1,16,end,256});auto ms=slice(mask,Shape{start,0},Shape{end,nk});
   auto o=fast::scaled_dot_product_attention(qs,k,v,0.0625f,"array",ms);eval(o);out.push_back(o);
  }return concatenate(out,2);
 };
 for(int stress=0;stress<2;stress++){
  if(stress){q=multiply(q,array(3.0f));k=multiply(k,array(3.0f));eval(q,k);}
  auto ref=run(0);eval(ref);
  for(int arm=1;arm<5;arm++)for(int causal=0;causal<(arm>=2?2:1);causal++){
   auto out=run(arm,causal);eval(out);auto err=max(abs(subtract(ref,out)));auto rel=sqrt(divide(sum(square(subtract(ref,out))),sum(square(ref))));eval(err,rel);
   float e=err.item<float>(),r=rel.item<float>();
   std::cout<<"{\"check\":\""<<names[arm]<<"\",\"stress\":"<<stress<<",\"causal\":"<<causal<<",\"max_abs\":"<<e<<",\"rel_l2\":"<<r<<"}"<<std::endl;
   if(!std::isfinite(e)||!std::isfinite(r)||e>(stress?1e-4f:1e-5f)||r>1e-5f)return 2;
  }
 }
 // Restore precisely the original values, avoiding scale/unscale rounding.
 random::seed(20260908);q=transpose(random::normal({1,nq,16,256}),{0,2,1,3});k=slice(random::normal({1,2,nk+256,256}),Shape{0,0,0,0},Shape{1,2,nk,256});eval(q,k);
 for(int arm=0;arm<5;arm++)for(int i=0;i<3;i++){auto o=run(arm);eval(o);}
 for(int round=0;round<6;round++){
  std::vector<int> order={0,1,2,3,4};std::rotate(order.begin(),order.begin()+(round%5),order.end());if(round%2)std::reverse(order.begin(),order.end());
  for(int arm:order){std::vector<double> ms;for(int rep=0;rep<3;rep++){
    auto t=std::chrono::steady_clock::now();auto o=run(arm);eval(o);ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
   }std::sort(ms.begin(),ms.end());std::cout<<"{\"round\":"<<round<<",\"arm\":\""<<names[arm]<<"\",\"median_ms\":"<<ms[1]<<",\"min_ms\":"<<ms[0]<<",\"max_ms\":"<<ms[2]<<"}"<<std::endl;
  }
 }
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
