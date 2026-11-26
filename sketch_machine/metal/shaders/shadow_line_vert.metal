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

struct ShadowLineUniforms {
  float4x4 svp;
  float4 line_info;
};

struct shadow_line_vert_out {
  float4 position [[position]];
};

struct shadow_line_vert_in {
  float3 in_pos [[attribute(0)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float4& clip_position, constant ShadowLineUniforms& uniforms,
    thread float3& vertex_position) {
  clip_position = uniforms.svp * float4(vertex_position, 1.0);
}

vertex shadow_line_vert_out
shadow_line_vert(shadow_line_vert_in stage_input [[stage_in]],
                 constant ShadowLineUniforms& uniforms [[buffer(0)]]) {
  shadow_line_vert_out stage_output = {};
  shade_vertex(stage_output.position, uniforms, stage_input.in_pos);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(ShadowLineUniforms) == 80, "Uniform size");
static_assert(__builtin_offsetof(ShadowLineUniforms, svp) == 0, "svp offset");
static_assert(__builtin_offsetof(ShadowLineUniforms, line_info) == 64,
              "line_info offset");
