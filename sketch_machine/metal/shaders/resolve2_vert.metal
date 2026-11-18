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
#include <simd/simd.h>

#include <metal_stdlib>

using namespace metal;

struct Resolve2Uniforms {
  float4 color;
  float4 extra;
};

struct resolve2_vert_out {
  float2 v_uv [[user(locn0)]];
  float4 position [[position]];
};

vertex resolve2_vert_out resolve2_vert(constant Resolve2Uniforms& uniforms
                                       [[buffer(0)]],
                                       uint vertex_index [[vertex_id]]) {
  resolve2_vert_out stage_output = {};
  float2 triangle_corner =
      float2(float((int(vertex_index) << 1) & 2), float(int(vertex_index) & 2));
  stage_output.v_uv = triangle_corner;
  stage_output.position =
      float4((triangle_corner * 2.0) - float2(1.0), uniforms.extra.x, 1.0);
  return stage_output;
}

static_assert(sizeof(Resolve2Uniforms) == 32, "Uniform size");
static_assert(__builtin_offsetof(Resolve2Uniforms, color) == 0, "color offset");
static_assert(__builtin_offsetof(Resolve2Uniforms, extra) == 16,
              "extra offset");
