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

struct PickUniforms {
  float4x4 mvp;
  float4x4 model;
  float mesh_index;
  char padding0[12];
  float4 line_info;
};

struct pick_frag_out {
  float4 f_out [[color(0)]];
  float f_prim [[color(1)]];
};

struct pick_frag_in {
  float3 v_world [[user(locn0)]];
};

fragment pick_frag_out pick_frag(pick_frag_in stage_input [[stage_in]],
                                 constant PickUniforms& uniforms [[buffer(0)]],
                                 uint primitive_index [[primitive_id]]) {
  pick_frag_out stage_output = {};
  stage_output.f_out = float4(stage_input.v_world, uniforms.mesh_index);
  stage_output.f_prim = float(int(primitive_index));
  return stage_output;
}

static_assert(sizeof(PickUniforms) == 160, "Uniform size");
static_assert(__builtin_offsetof(PickUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(PickUniforms, model) == 64, "model offset");
static_assert(__builtin_offsetof(PickUniforms, mesh_index) == 128,
              "mesh_index offset");
static_assert(__builtin_offsetof(PickUniforms, line_info) == 144,
              "line_info offset");
