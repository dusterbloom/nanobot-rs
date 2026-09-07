import Foundation
import Metal
let device = MTLCreateSystemDefaultDevice()!
let count = 16 * 1024 * 1024
let bytes = count * 16
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void stream_copy(device const float4* a [[buffer(0)]], device float4* b [[buffer(1)]], uint i [[thread_position_in_grid]]) { b[i] = a[i]; }
"""
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "stream_copy")!)
let queue = device.makeCommandQueue()!
let input = device.makeBuffer(length: bytes, options: .storageModeShared)!
let output = device.makeBuffer(length: bytes, options: .storageModeShared)!
input.contents().initializeMemory(as: UInt8.self, repeating: 13, count: bytes)
output.contents().initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
var seconds: [Double] = []
for iteration in 0..<25 {
 let command = queue.makeCommandBuffer()!
 let encoder = command.makeComputeCommandEncoder()!
 encoder.setComputePipelineState(pipeline)
 encoder.setBuffer(input, offset: 0, index: 0)
 encoder.setBuffer(output, offset: 0, index: 1)
 encoder.dispatchThreads(MTLSize(width: count,height: 1,depth: 1), threadsPerThreadgroup: MTLSize(width: 256,height: 1,depth: 1))
 encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
 if command.status != .completed { fatalError("GPU command failed: \(String(describing: command.error))") }
 if iteration >= 5 { seconds.append(command.gpuEndTime-command.gpuStartTime) }
}
precondition(output.contents().load(as: UInt8.self) == 13)
precondition(output.contents().advanced(by: bytes-1).load(as: UInt8.self) == 13)
seconds.sort()
let median = (seconds[9]+seconds[10])/2
let result: [String: Any] = ["device":device.name,"buffer_bytes":bytes,"traffic_bytes_per_dispatch":bytes*2,"median_gpu_seconds":median,"decimal_GB_s":Double(bytes*2)/median/1e9,"timed_samples":seconds,"note":"Streaming read+write copy, 256MiB per buffer; operation-specific bandwidth, not a measured inference roofline."]
let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted,.sortedKeys])
FileHandle.standardOutput.write(json)
