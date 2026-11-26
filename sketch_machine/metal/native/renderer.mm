/*
 * Mule Create
 * Copyright (C) 2026 Felix Barthel
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of Mule Create.
 *
 * Mule Create is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, version 3 of the License.
 *
 * Mule Create is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Mule Create. If not, see <https://www.gnu.org/licenses/>.
 */

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

// A Qt NSView receives all input. MetalKit supplies only the drawable-backed child view.
@interface MuleMetalView : MTKView
@end
@implementation MuleMetalView
- (NSView *)hitTest:(NSPoint)point {
  return nil;
}

@end

static void drawView(void *backend, MTKView *view);
@interface MuleMetalDelegate : NSObject <MTKViewDelegate>
@property(assign) void *backend;
@end
@implementation MuleMetalDelegate
- (void)drawInMTKView:(MTKView *)view {
  drawView(self.backend, view);
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
}

@end

@interface MulePipeline : NSObject
@property(strong) id<MTLRenderPipelineState> state;
@property(strong) id<MTLDepthStencilState> depth;
@end
@implementation MulePipeline
@end

@interface MuleGeometry : NSObject
@property(strong) id<MTLAccelerationStructure> acceleration;
@property(strong) id<MTLBuffer> vertices;
@property(strong) id<MTLBuffer> indices;
@end
@implementation MuleGeometry
@end

static thread_local std::string errorText;
static void require(bool value, NSString *message) {
  if (!value)
    throw std::runtime_error(message.UTF8String ?: "Metal operation failed");
}

template <class F>
static auto protect(F fn) -> decltype(fn()) {
  @autoreleasepool {
    try {
      errorText.clear();
      return fn();
    } catch (const std::exception &e) {
      errorText = e.what();
      return {};
    }
  }
}

static id object(void *p) { return (__bridge id)p; }
static void *owned(id obj) { return obj ? (void *)CFBridgingRetain(obj) : nullptr; }
static NSDictionary *json(const char *data) {
  return [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:data length:strlen(data)]
                                         options:0
                                           error:nil];
}

struct Feedback {
  std::mutex mutex;
  std::string error;
};
struct Frame {
  id<MTL4CommandAllocator> allocator;
  id<MTL4CommandBuffer> commands;
  id<MTLResidencySet> residency;
  NSMutableArray *held;
  NSMutableArray<id<MTL4ArgumentTable>> *tables;
  NSMutableArray<id<MTLBuffer>> *constants;
  size_t tableUsed = 0, constantUsed = 0;
  uint64_t complete = 0;
};

struct Backend {
  id<MTLDevice> device;
  id<MTL4CommandQueue> queue;
  id<MTL4Compiler> compiler;
  id<MTLSharedEvent> done;
  id<MTL4RenderCommandEncoder> render;
  id<MTL4ComputeCommandEncoder> compute;
  id<MTLLibrary> builtins;
  id<MTLRenderPipelineState> composite, copy;
  id<MTLSamplerState> linear;
  id<MTLTexture> output;
  id<MTLComputePipelineState> rayPipeline;
  id<MTLAccelerationStructure> rayScene;
  NSArray<MuleGeometry *> *rayGeometry;
  MuleMetalView *view;
  MuleMetalDelegate *delegate;
  std::array<Frame, 3> frames;
  Frame *frame = nullptr;
  uint64_t serial = 0, drawCount = 0, rtQueries = 0, presentedFrames = 0;
  std::shared_ptr<Feedback> feedback = std::make_shared<Feedback>();

  Backend(void *parent, const char *libraryPath) {
    if (@available(macOS 27.0, *)) {
    } else {
      require(false, @"Native Metal rendering requires macOS 27 or later");
    }

    device = MTLCreateSystemDefaultDevice();
    require(device && [device supportsFamily:MTLGPUFamilyMetal4],
            @"This GPU does not support Metal 4");
    NSError *error = nil;
    MTL4CommandQueueDescriptor *qd = [MTL4CommandQueueDescriptor new];
    qd.label = @"Mule native Metal 4 scene queue";
    queue = [device newMTL4CommandQueueWithDescriptor:qd error:&error];
    compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    builtins = [device newLibraryWithURL:[NSURL fileURLWithPath:@(libraryPath)] error:&error];
    done = [device newSharedEvent];
    require(queue && compiler && builtins && done, error.localizedDescription);
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.supportArgumentBuffers = YES;
    linear = [device newSamplerStateWithDescriptor:sd];
    for (auto &f : frames) {
      f.allocator = [device newCommandAllocator];
      f.commands = [device newCommandBuffer];
      MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
      rd.initialCapacity = 256;
      f.residency = [device newResidencySetWithDescriptor:rd error:&error];
      f.held = [NSMutableArray new];
      f.tables = [NSMutableArray new];
      f.constants = [NSMutableArray new];
      require(f.allocator && f.commands && f.residency, error.localizedDescription);
      [queue addResidencySet:f.residency];
    }

    if (parent) {
      NSView *host = (__bridge NSView *)parent;
      view = [[MuleMetalView alloc] initWithFrame:host.bounds device:device];
      view.paused = YES;
      view.enableSetNeedsDisplay = NO;
      view.autoResizeDrawable = NO;
      view.framebufferOnly = YES;
      view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
      view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
      delegate = [MuleMetalDelegate new];
      delegate.backend = this;
      view.delegate = delegate;
      [host addSubview:view positioned:NSWindowBelow relativeTo:nil];
      [queue addResidencySet:((CAMetalLayer *)view.layer).residencySet];
    }
  }

  ~Backend() {
    try {
      submit(true);
    } catch (...) {
    }

    for (auto &f : frames) {
      if (f.complete)
        [done waitUntilSignaledValue:f.complete timeoutMS:10000];

      [queue removeResidencySet:f.residency];
    }

    if (view)
      [queue removeResidencySet:((CAMetalLayer *)view.layer).residencySet];

    [view removeFromSuperview];
  }

  Frame &active() {
    {
      std::lock_guard<std::mutex> lock(feedback->mutex);
      require(feedback->error.empty(), @(feedback->error.c_str()));
    }

    if (!frame) {
      frame = &frames[serial % frames.size()];
      require(!frame->complete || [done waitUntilSignaledValue:frame->complete timeoutMS:10000],
              @"Metal frame timed out");
      [frame->allocator reset];
      [frame->held removeAllObjects];
      [frame->residency removeAllAllocations];
      frame->tableUsed = frame->constantUsed = 0;
      [frame->commands beginCommandBufferWithAllocator:frame->allocator];
    }

    return *frame;
  }

  void keep(id obj) {
    if (obj)
      [active().held addObject:obj];
  }

  void use(id<MTLAllocation> resource) {
    if (resource) {
      [active().residency addAllocation:resource];
      keep(resource);
    }
  }

  void end() {
    if (render) {
      [render endEncoding];
      render = nil;
    }

    if (compute) {
      [compute endEncoding];
      compute = nil;
    }
  }

  id<MTL4ComputeCommandEncoder> blit() {
    if (!compute) {
      end();
      compute = [active().commands computeCommandEncoder];
      [compute barrierAfterQueueStages:MTLStageAll
                          beforeStages:MTLStageAll
                     visibilityOptions:MTL4VisibilityOptionDevice];
    } else {
      [compute barrierAfterEncoderStages:MTLStageBlit
                     beforeEncoderStages:MTLStageBlit
                       visibilityOptions:MTL4VisibilityOptionDevice];
    }

    return compute;
  }

  id<MTL4ArgumentTable> arguments() {
    Frame &f = active();

    if (f.tableUsed == f.tables.count) {
      MTL4ArgumentTableDescriptor *d = [MTL4ArgumentTableDescriptor new];
      d.maxBufferBindCount = 31;
      d.maxTextureBindCount = 31;
      d.maxSamplerStateBindCount = 16;
      d.initializeBindings = YES;
      NSError *e = nil;
      id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:d error:&e];
      require(table != nil, e.localizedDescription);
      [f.tables addObject:table];
    }

    return f.tables[f.tableUsed++];
  }

  MTLGPUAddress constants(const void *bytes, size_t size) {
    require(size <= (1 << 20), @"Uniform block exceeds the native constant-buffer chunk size");
    Frame &f = active();
    size_t offset = (f.constantUsed + 255) & ~size_t(255), chunk = offset / (1 << 20);

    if (offset % (1 << 20) + size > (1 << 20)) {
      chunk++;
      offset = chunk * (1 << 20);
    }
    while (chunk >= f.constants.count)
      [f.constants addObject:[device newBufferWithLength:1 << 20
                                                 options:MTLResourceStorageModeShared]];
    id<MTLBuffer> buffer = f.constants[chunk];
    use(buffer);
    memcpy((char *)buffer.contents + offset % (1 << 20), bytes, size);
    f.constantUsed = offset + size;
    return buffer.gpuAddress + offset % (1 << 20);
  }

  void submit(bool wait, id<CAMetalDrawable> drawable = nil) {
    if (!frame) {
      if (wait && serial)
        require([done waitUntilSignaledValue:serial timeoutMS:10000],
                @"Metal command submission timed out");

      return;
    }

    end();
    [frame->residency commit];
    [frame->commands endCommandBuffer];

    if (drawable)
      [queue waitForDrawable:drawable];

    MTL4CommitOptions *opts = [MTL4CommitOptions new];
    auto state = feedback;
    [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
      if (fb.error) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->error = fb.error.localizedDescription.UTF8String;
      }
    }];
    id<MTL4CommandBuffer> list[] = {frame->commands};
    [queue commit:list count:1 options:opts];
    frame->complete = ++serial;
    [queue signalEvent:done value:serial];

    if (drawable) {
      [queue signalDrawable:drawable];
      [drawable present];
      presentedFrames++;
    }

    frame = nullptr;

    if (wait)
      require([done waitUntilSignaledValue:serial timeoutMS:10000],
              @"Metal command submission timed out");
  }

  void pass(NSArray *colors, id<MTLTexture> depth, const double *clear, int flags, double z) {
    end();
    MTL4RenderPassDescriptor *d = [MTL4RenderPassDescriptor new];
    for (NSUInteger i = 0; i < colors.count; i++) {
      id<MTLTexture> t = colors[i];
      use(t);
      d.colorAttachments[i].texture = t;
      d.colorAttachments[i].loadAction = flags & 1 ? MTLLoadActionClear : MTLLoadActionLoad;
      d.colorAttachments[i].storeAction = MTLStoreActionStore;
      d.colorAttachments[i].clearColor = MTLClearColorMake(clear[0], clear[1], clear[2], clear[3]);
    }

    if (depth) {
      use(depth);
      d.depthAttachment.texture = depth;
      d.depthAttachment.loadAction = flags & 2 ? MTLLoadActionClear : MTLLoadActionLoad;
      d.depthAttachment.storeAction = MTLStoreActionStore;
      d.depthAttachment.clearDepth = z;
    }

    render = [active().commands renderCommandEncoderWithDescriptor:d];
    require(render != nil, @"Could not open Metal render pass");
    [render barrierAfterQueueStages:MTLStageAll
                       beforeStages:MTLStageAll
                  visibilityOptions:MTL4VisibilityOptionDevice];
  }

  id<MTLRenderPipelineState> builtin(NSString *fragment, MTLPixelFormat format) {
    MTL4RenderPipelineDescriptor *d = [MTL4RenderPipelineDescriptor new];
    MTL4LibraryFunctionDescriptor *v = [MTL4LibraryFunctionDescriptor new],
                                  *f = [MTL4LibraryFunctionDescriptor new];
    v.library = f.library = builtins;
    v.name = @"mule_vertex";
    f.name = fragment;
    d.vertexFunctionDescriptor = v;
    d.fragmentFunctionDescriptor = f;
    d.colorAttachments[0].pixelFormat = format;
    NSError *e = nil;
    id<MTLRenderPipelineState> p = [compiler newRenderPipelineStateWithDescriptor:d
                                                              compilerTaskOptions:nil
                                                                            error:&e];
    require(p != nil, e.localizedDescription);
    return p;
  }

  void fullscreen(id<MTLTexture> target, id<MTLRenderPipelineState> pipeline, id<MTLTexture> a,
                  id<MTLTexture> b) {
    double clear[] = {0, 0, 0, 0};
    pass(@[ target ], nil, clear, 1, 1);
    use(a);
    use(b);
    keep(pipeline);
    id<MTL4ArgumentTable> table = arguments();
    [table setTexture:a.gpuResourceID atIndex:0];
    [table setTexture:b.gpuResourceID atIndex:1];
    [table setSamplerState:linear.gpuResourceID atIndex:0];
    [render setRenderPipelineState:pipeline];
    [render setArgumentTable:table atStages:MTLRenderStageFragment];
    [render setViewport:MTLViewport{0, 0, double(target.width), double(target.height), 0, 1}];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    drawCount++;
  }

  id<MTLAccelerationStructure> buildAcceleration(MTL4AccelerationStructureDescriptor *desc) {
    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:desc];
    id<MTLAccelerationStructure> structure =
        [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch = [device newBufferWithLength:MAX(256, sizes.buildScratchBufferSize)
                                                options:MTLResourceStorageModePrivate];
    require(structure && scratch, @"Could not allocate MetalRT acceleration structure");
    use(structure);
    use(scratch);
    [blit() buildAccelerationStructure:structure
                            descriptor:desc
                         scratchBuffer:MTL4BufferRange{scratch.gpuAddress, scratch.length}];
    // A build may consume preceding geometry uploads, and the next build/dispatch consumes its
    // output.
    [compute barrierAfterEncoderStages:MTLStageAccelerationStructure
                   beforeEncoderStages:MTLStageAccelerationStructure | MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
    return structure;
  }
};

static void drawView(void *backend, MTKView *view) {
  auto b = (Backend *)backend;
  id<CAMetalDrawable> drawable = view.currentDrawable;

  if (drawable && b->output) {
    if (!b->copy)
      b->copy = b->builtin(@"mule_copy_fragment", MTLPixelFormatBGRA8Unorm);

    b->fullscreen(drawable.texture, b->copy, b->output, b->output);
    b->submit(false, drawable);
  }
}

extern "C" const char *mc_error() { return errorText.c_str(); }
// Probe the GPU selected on this machine before choosing the viewport's widget type.
// This creates no windows and does not rely on the machine that built Mule.app.
extern "C" const char *mc_probe(const char *builtinPath, const char *scenePath) {
  static thread_local std::string result;
  return protect([&]() -> const char * {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    bool supported = false;
    NSString *reason = @"Native Metal rendering requires macOS 27 or later";

    if (@available(macOS 27.0, *)) {
      if (!device) {
        reason = @"No Metal GPU is available";
      } else if (![device supportsFamily:MTLGPUFamilyMetal4]) {
        reason = @"This GPU does not support Metal 4";
      } else {
        NSError *error = nil;
        id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                                                error:&error];
        id<MTL4CommandQueue> queue =
            [device newMTL4CommandQueueWithDescriptor:[MTL4CommandQueueDescriptor new]
                                                error:&error];
        id<MTLLibrary> builtin = [device newLibraryWithURL:[NSURL fileURLWithPath:@(builtinPath)]
                                                     error:&error];
        id<MTLLibrary> scene = [device newLibraryWithURL:[NSURL fileURLWithPath:@(scenePath)]
                                                   error:&error];
        supported = compiler && queue && builtin && scene;
        reason =
            supported ? @"" : (error.localizedDescription ?: @"Native Metal initialization failed");
      }
    }

    NSDictionary *d = @{
      @"available" : @(supported),
      @"reason" : reason,
      @"device" : device.name ?: @"",
      @"metalrt" : @(device && device.supportsRaytracing)
    };
    NSData *data = [NSJSONSerialization dataWithJSONObject:d options:0 error:nil];
    result.assign((const char *)data.bytes, data.length);
    return result.c_str();
  });
}

extern "C" void *mc_create(void *parent, const char *library) {
  return protect([&]() -> void * { return new Backend(parent, library); });
}

extern "C" void mc_destroy(void *p) { delete static_cast<Backend *>(p); }
extern "C" void mc_release(void *p) {
  if (p) {
    id value = CFBridgingRelease(p);
    (void)value;
  }
}

extern "C" const char *mc_info(void *p) {
  static thread_local std::string result;
  return protect([&]() -> const char * {
    auto b = (Backend *)p;
    NSDictionary *d = @{
      @"device" : b->device.name,
      @"metalrt" : @(b->device.supportsRaytracing),
      @"draws" : @(b->drawCount),
      @"ray_queries" : @(b->rtQueries),
      @"presented_frames" : @(b->presentedFrames),
      @"frames" : @(b->serial)
    };
    NSData *data = [NSJSONSerialization dataWithJSONObject:d options:0 error:nil];
    result.assign((const char *)data.bytes, data.length);
    return result.c_str();
  });
}

extern "C" int mc_submit(void *p, int wait) {
  return protect([&] {
    ((Backend *)p)->submit(wait);
    return 1;
  });
}

extern "C" void *mc_buffer(void *p, const void *bytes, size_t size) {
  return protect([&]() -> void * {
    auto b = (Backend *)p;
    id<MTLBuffer> buffer = bytes ? [b->device newBufferWithBytes:bytes
                                                          length:size
                                                         options:MTLResourceStorageModeShared]
                                 : [b->device newBufferWithLength:size
                                                          options:MTLResourceStorageModeShared];
    require(buffer != nil, @"Could not allocate Metal buffer");
    return owned(buffer);
  });
}

extern "C" int mc_write_buffer(void *p, void *buffer, const void *bytes, size_t offset,
                               size_t size) {
  return protect([&] {
    auto b = (Backend *)p;
    id<MTLBuffer> dst = object(buffer),
                  src = [b->device newBufferWithBytes:bytes
                                               length:size
                                              options:MTLResourceStorageModeShared];
    require(offset + size <= dst.length, @"Buffer write out of bounds");
    b->use(src);
    b->use(dst);
    [b->blit() copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:offset size:size];
    return 1;
  });
}

extern "C" int mc_read_buffer(void *p, void *buffer, void *bytes, size_t offset, size_t size) {
  return protect([&] {
    auto b = (Backend *)p;
    id<MTLBuffer> src = object(buffer);
    require(offset + size <= src.length, @"Buffer read out of bounds");
    b->submit(true);
    memcpy(bytes, (char *)src.contents + offset, size);
    return 1;
  });
}

extern "C" void *mc_texture(void *p, int width, int height, int format, int samples, int mips) {
  return protect([&]() -> void * {
    auto b = (Backend *)p;
    MTLTextureDescriptor *d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)format
                                                           width:width
                                                          height:height
                                                       mipmapped:mips];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModePrivate;

    if (samples > 1) {
      d.textureType = MTLTextureType2DMultisample;
      d.sampleCount = samples;
      d.mipmapLevelCount = 1;
    }

    id<MTLTexture> t = [b->device newTextureWithDescriptor:d];
    require(t != nil, @"Could not allocate Metal texture");
    return owned(t);
  });
}

extern "C" int mc_write_texture(void *p, void *tex, const void *bytes, int x, int y, int w, int h,
                                int level, int bpp) {
  return protect([&] {
    auto b = (Backend *)p;
    size_t row = (w * bpp + 255) & ~size_t(255);
    id<MTLBuffer> src = [b->device newBufferWithLength:row * h
                                               options:MTLResourceStorageModeShared];
    for (int k = 0; k < h; k++)
      memcpy((char *)src.contents + k * row, (const char *)bytes + k * w * bpp, w * bpp);
    id<MTLTexture> dst = object(tex);
    b->use(src);
    b->use(dst);
    [b->blit() copyFromBuffer:src
                 sourceOffset:0
            sourceBytesPerRow:row
          sourceBytesPerImage:row * h
                   sourceSize:MTLSizeMake(w, h, 1)
                    toTexture:dst
             destinationSlice:0
             destinationLevel:level
            destinationOrigin:MTLOriginMake(x, y, 0)];
    return 1;
  });
}

extern "C" int mc_read_texture(void *p, void *tex, void *bytes, int x, int y, int w, int h,
                               int bpp) {
  return protect([&] {
    auto b = (Backend *)p;
    id<MTLTexture> src = object(tex);
    size_t row = (w * bpp + 255) & ~size_t(255);
    id<MTLBuffer> dst = [b->device newBufferWithLength:row * h
                                               options:MTLResourceStorageModeShared];
    b->use(src);
    b->use(dst);
    [b->blit() copyFromTexture:src
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(x, y, 0)
                      sourceSize:MTLSizeMake(w, h, 1)
                        toBuffer:dst
               destinationOffset:0
          destinationBytesPerRow:row
        destinationBytesPerImage:row * h];
    b->submit(true);
    for (int k = 0; k < h; k++)
      memcpy((char *)bytes + k * w * bpp, (char *)dst.contents + k * row, w * bpp);
    return 1;
  });
}

extern "C" int mc_mipmap(void *p, void *tex) {
  return protect([&] {
    auto b = (Backend *)p;
    id<MTLTexture> t = object(tex);
    b->use(t);
    [b->blit() generateMipmapsForTexture:t];
    return 1;
  });
}

extern "C" int mc_copy_texture(void *p, void *src, void *dst) {
  return protect([&] {
    auto b = (Backend *)p;
    id<MTLTexture> s = object(src), d = object(dst);
    b->use(s);
    b->use(d);
    [b->blit() copyFromTexture:s toTexture:d];
    return 1;
  });
}

extern "C" void *mc_sampler(void *p, int min, int mag, int mip, int rx, int ry) {
  return protect([&]() -> void * {
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.minFilter = (MTLSamplerMinMagFilter)min;
    d.magFilter = (MTLSamplerMinMagFilter)mag;
    d.mipFilter = (MTLSamplerMipFilter)mip;
    d.sAddressMode = rx ? MTLSamplerAddressModeRepeat : MTLSamplerAddressModeClampToEdge;
    d.tAddressMode = ry ? MTLSamplerAddressModeRepeat : MTLSamplerAddressModeClampToEdge;
    d.supportArgumentBuffers = YES;
    return owned([((Backend *)p)->device newSamplerStateWithDescriptor:d]);
  });
}

extern "C" void *mc_library(void *p, const void *data, size_t size) {
  return protect([&]() -> void * {
    dispatch_data_t d = dispatch_data_create(data, size, dispatch_get_global_queue(0, 0),
                                             DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *e = nil;
    id<MTLLibrary> l = [((Backend *)p)->device newLibraryWithData:d error:&e];
    require(l != nil, e.localizedDescription);
    return owned(l);
  });
}

extern "C" void *mc_pipeline(void *p, void *vlib, void *flib, const char *description) {
  return protect([&]() -> void * {
    auto b = (Backend *)p;
    NSDictionary *j = json(description);
    MTL4RenderPipelineDescriptor *d = [MTL4RenderPipelineDescriptor new];
    MTL4LibraryFunctionDescriptor *v = [MTL4LibraryFunctionDescriptor new],
                                  *f = [MTL4LibraryFunctionDescriptor new];
    v.library = object(vlib);
    f.library = object(flib);
    v.name = j[@"vs"];
    f.name = j[@"fs"];
    d.vertexFunctionDescriptor = v;
    d.fragmentFunctionDescriptor = f;
    d.rasterSampleCount = [j[@"samples"] unsignedIntegerValue];
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    for (NSArray *a in j[@"attributes"]) {
      int loc = [a[0] intValue];
      vd.attributes[loc].bufferIndex = 16 + [a[1] intValue];
      vd.attributes[loc].format = (MTLVertexFormat)[a[2] intValue];
      vd.attributes[loc].offset = [a[3] unsignedIntegerValue];
    }

    int index = 16;
    for (NSArray *a in j[@"bindings"]) {
      vd.layouts[index].stride = [a[0] unsignedIntegerValue];
      vd.layouts[index].stepFunction =
          [a[1] boolValue] ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
      vd.layouts[index].stepRate = 1;
      index++;
    }

    d.vertexDescriptor = vd;
    int i = 0;
    for (NSNumber *format in j[@"formats"]) {
      auto a = d.colorAttachments[i++];
      a.pixelFormat = (MTLPixelFormat)format.unsignedIntegerValue;
      a.writeMask = (MTLColorWriteMask)[j[@"mask"] unsignedIntegerValue];
      NSArray *blend = j[@"blend"];

      if (blend.count) {
        a.blendingState = MTL4BlendStateEnabled;
        a.sourceRGBBlendFactor = (MTLBlendFactor)[blend[0] intValue];
        a.destinationRGBBlendFactor = (MTLBlendFactor)[blend[1] intValue];
        a.sourceAlphaBlendFactor = (MTLBlendFactor)[blend[2] intValue];
        a.destinationAlphaBlendFactor = (MTLBlendFactor)[blend[3] intValue];
      }
    }

    NSError *e = nil;
    MulePipeline *out = [MulePipeline new];
    out.state = [b->compiler newRenderPipelineStateWithDescriptor:d
                                              compilerTaskOptions:nil
                                                            error:&e];
    require(out.state != nil, e.localizedDescription);
    MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = (MTLCompareFunction)[j[@"compare"] intValue];
    ds.depthWriteEnabled = [j[@"depth_write"] boolValue];
    out.depth = [b->device newDepthStencilStateWithDescriptor:ds];
    return owned(out);
  });
}

extern "C" int mc_pass(void *p, void **colors, int count, void *depth, const double *clear,
                       int flags, double z) {
  return protect([&] {
    NSMutableArray *list = [NSMutableArray new];
    for (int i = 0; i < count; i++) [list addObject:object(colors[i])];
    ((Backend *)p)->pass(list, object(depth), clear, flags, z);
    return 1;
  });
}

extern "C" int mc_end_pass(void *p) {
  return protect([&] {
    ((Backend *)p)->end();
    return 1;
  });
}

struct VertexBinding {
  void *buffer;
  uint64_t offset;
};
struct TextureBinding {
  int stage, texture, sampler;
  void *image;
  void *state;
};
extern "C" int mc_draw(void *p, void *pipeline, const VertexBinding *vertices, int vertexCount,
                       const TextureBinding *textures, int textureCount, const void *uniforms,
                       size_t uniformSize, const double *viewport, const int *scissor, int topology,
                       int count, int first, int instances, void *indices) {
  return protect([&] {
    auto b = (Backend *)p;
    require(b->render != nil, @"Draw outside render pass");
    MulePipeline *pipe = object(pipeline);
    b->keep(pipe);
    id<MTL4ArgumentTable> vt = b->arguments(), ft = b->arguments();

    if (uniformSize) {
      MTLGPUAddress address = b->constants(uniforms, uniformSize);
      [vt setAddress:address atIndex:0];
      [ft setAddress:address atIndex:0];
    }

    for (int i = 0; i < vertexCount; i++) {
      id<MTLBuffer> buf = object(vertices[i].buffer);
      b->use(buf);
      [vt setAddress:buf.gpuAddress + vertices[i].offset atIndex:16 + i];
    }

    for (int i = 0; i < textureCount; i++) {
      auto &t = textures[i];
      id<MTLTexture> tex = object(t.image);
      id<MTLSamplerState> sampler = object(t.state);
      b->use(tex);
      b->keep(sampler);
      id<MTL4ArgumentTable> table = t.stage ? ft : vt;

      if (t.texture >= 0)
        [table setTexture:tex.gpuResourceID atIndex:t.texture];

      if (t.sampler >= 0)
        [table setSamplerState:sampler.gpuResourceID atIndex:t.sampler];
    }

    [b->render setArgumentTable:vt atStages:MTLRenderStageVertex];
    [b->render setArgumentTable:ft atStages:MTLRenderStageFragment];
    [b->render setRenderPipelineState:pipe.state];
    [b->render setDepthStencilState:pipe.depth];
    [b->render setCullMode:MTLCullModeNone];
    [b->render setViewport:MTLViewport{viewport[0], viewport[1], viewport[2], viewport[3], 0, 1}];
    [b->render setScissorRect:MTLScissorRect{(NSUInteger)scissor[0], (NSUInteger)scissor[1],
                                             (NSUInteger)scissor[2], (NSUInteger)scissor[3]}];
    if (indices) {
      id<MTLBuffer> ib = object(indices);
      b->use(ib);
      [b->render drawIndexedPrimitives:(MTLPrimitiveType)topology
                            indexCount:count
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:ib.gpuAddress + first * 4
                     indexBufferLength:ib.length - first * 4
                         instanceCount:instances];
    } else
      [b->render drawPrimitives:(MTLPrimitiveType)topology
                    vertexStart:first
                    vertexCount:count
                  instanceCount:instances];
    b->drawCount++;
    return 1;
  });
}

extern "C" int mc_present(void *p, void *scene, void *overlay, int width, int height) {
  return protect([&] {
    auto b = (Backend *)p;

    if (!b->output || b->output.width != width || b->output.height != height) {
      MTLTextureDescriptor *d =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                             width:width
                                                            height:height
                                                         mipmapped:NO];
      d.storageMode = MTLStorageModePrivate;
      d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
      b->output = [b->device newTextureWithDescriptor:d];
    }

    if (!b->composite)
      b->composite = b->builtin(@"mule_fragment", MTLPixelFormatRGBA8Unorm);

    b->fullscreen(b->output, b->composite, object(scene), object(overlay));
    b->end();

    if (b->view) {
      b->view.frame = b->view.superview.bounds;
      b->view.drawableSize = CGSizeMake(width, height);
      [b->view draw];
    }

    b->submit(false);
    return 1;
  });
}

extern "C" void *mc_output(void *p) { return (__bridge void *)((Backend *)p)->output; }

extern "C" void *mc_rt_geometry(void *p, void *vertices, void *indices, size_t triangleCount) {
  return protect([&]() -> void * {
    auto b = (Backend *)p;
    require(b->device.supportsRaytracing, @"MetalRT is unavailable on this device");
    b->end();
    MuleGeometry *geo = [MuleGeometry new];
    geo.vertices = object(vertices);
    geo.indices = object(indices);
    MTL4AccelerationStructureTriangleGeometryDescriptor *t =
        [MTL4AccelerationStructureTriangleGeometryDescriptor new];
    t.vertexBuffer = MTL4BufferRange{geo.vertices.gpuAddress, geo.vertices.length};
    t.vertexStride = 12;
    t.vertexFormat = MTLAttributeFormatFloat3;
    t.indexBuffer = MTL4BufferRange{geo.indices.gpuAddress, geo.indices.length};
    t.indexType = MTLIndexTypeUInt32;
    t.triangleCount = triangleCount;
    t.opaque = YES;
    MTL4PrimitiveAccelerationStructureDescriptor *d =
        [MTL4PrimitiveAccelerationStructureDescriptor new];
    d.geometryDescriptors = @[ t ];
    b->use(geo.vertices);
    b->use(geo.indices);
    geo.acceleration = b->buildAcceleration(d);
    return owned(geo);
  });
}

extern "C" int mc_rt_scene(void *p, void **geometries, const float *transforms, int count) {
  return protect([&] {
    auto b = (Backend *)p;
    b->end();
    id<MTLBuffer> instances = [b->device
        newBufferWithLength:count * sizeof(MTLIndirectAccelerationStructureInstanceDescriptor)
                    options:MTLResourceStorageModeShared];
    auto out = (MTLIndirectAccelerationStructureInstanceDescriptor *)instances.contents;
    NSMutableArray *held = [NSMutableArray new];
    for (int i = 0; i < count; i++) {
      MuleGeometry *geo = object(geometries[i]);
      [held addObject:geo];
      b->use(geo.acceleration);
      memcpy(&out[i].transformationMatrix, transforms + i * 12, 48);
      out[i].mask = 0xff;
      out[i].options = MTLAccelerationStructureInstanceOptionOpaque;
      out[i].userID = i;
      out[i].accelerationStructureID = geo.acceleration.gpuResourceID;
    }

    MTL4InstanceAccelerationStructureDescriptor *d =
        [MTL4InstanceAccelerationStructureDescriptor new];
    d.instanceDescriptorBuffer = MTL4BufferRange{instances.gpuAddress, instances.length};
    d.instanceCount = count;
    d.instanceDescriptorStride = sizeof(*out);
    d.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeIndirect;
    b->use(instances);
    b->rayGeometry = held;
    b->rayScene = b->buildAcceleration(d);
    return 1;
  });
}

extern "C" int mc_rt_trace(void *p, const void *rays, void *hits, int count) {
  return protect([&] {
    auto b = (Backend *)p;
    require(b->rayScene != nil, @"MetalRT scene is empty");
    b->end();

    if (!b->rayPipeline) {
      MTL4ComputePipelineDescriptor *d = [MTL4ComputePipelineDescriptor new];
      MTL4LibraryFunctionDescriptor *f = [MTL4LibraryFunctionDescriptor new];
      f.library = b->builtins;
      f.name = @"mule_ray_hits";
      d.computeFunctionDescriptor = f;
      NSError *e = nil;
      b->rayPipeline = [b->compiler newComputePipelineStateWithDescriptor:d
                                                      compilerTaskOptions:nil
                                                                    error:&e];
      require(b->rayPipeline != nil, e.localizedDescription);
    }

    id<MTLBuffer> input = [b->device newBufferWithBytes:rays
                                                 length:count * 32
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [b->device newBufferWithLength:count * 16
                                                  options:MTLResourceStorageModeShared];
    b->use(input);
    b->use(output);
    b->use(b->rayScene);
    for (MuleGeometry *g in b->rayGeometry) b->use(g.acceleration);
    id<MTL4ArgumentTable> table = b->arguments();
    [table setResource:b->rayScene.gpuResourceID atBufferIndex:0];
    [table setAddress:input.gpuAddress atIndex:1];
    [table setAddress:output.gpuAddress atIndex:2];
    uint32_t n = count;
    [table setAddress:b->constants(&n, sizeof(n)) atIndex:3];
    auto enc = b->blit();
    [enc setComputePipelineState:b->rayPipeline];
    [enc setArgumentTable:table];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(count, 64), 1, 1)];
    b->submit(true);
    memcpy(hits, output.contents, count * 16);
    b->rtQueries += count;
    return 1;
  });
}
