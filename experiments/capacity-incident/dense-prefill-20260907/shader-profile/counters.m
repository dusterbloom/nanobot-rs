#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(void) { @autoreleasepool {
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();
 NSMutableArray *sets=[NSMutableArray array];
 for(id<MTLCounterSet> set in device.counterSets){
  NSMutableArray *names=[NSMutableArray array];
  for(id<MTLCounter> counter in set.counters)[names addObject:counter.name];
  [sets addObject:@{@"name":set.name,@"counters":names}];
 }
 NSDictionary *result=@{@"device":device.name,@"counterSets":sets,@"maxThreadgroupMemoryLength":@(device.maxThreadgroupMemoryLength)};
 NSData *data=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];
 puts([[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding].UTF8String);
}return 0;}
