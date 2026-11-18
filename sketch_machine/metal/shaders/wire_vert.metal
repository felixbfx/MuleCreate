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

// Native Metal shader. Maintain this source directly; metal/build.py compiles
// it.
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <simd/simd.h>

#include <metal_stdlib>

using namespace metal;

struct WireUniforms {
  float4x4 mvp;
  float bias0;
  char padding0[12];
  float4 color;
  float4 line_info;
};

struct wire_vert_out {
  float4 position [[position]];
};

struct wire_vert_in {
  float3 in_pos [[attribute(0)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float4& clip_position, constant WireUniforms& uniforms,
    thread float3& vertex_position) {
  clip_position = uniforms.mvp * float4(vertex_position, 1.0);
  clip_position.z -= (uniforms.bias0 * clip_position.w);
}

vertex wire_vert_out wire_vert(wire_vert_in stage_input [[stage_in]],
                               constant WireUniforms& uniforms [[buffer(0)]]) {
  wire_vert_out stage_output = {};
  shade_vertex(stage_output.position, uniforms, stage_input.in_pos);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(WireUniforms) == 112, "Uniform size");
static_assert(__builtin_offsetof(WireUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(WireUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(WireUniforms, color) == 80, "color offset");
static_assert(__builtin_offsetof(WireUniforms, line_info) == 96,
              "line_info offset");
