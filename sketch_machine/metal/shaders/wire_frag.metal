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

struct WireUniforms {
  float4x4 mvp;
  float bias0;
  char padding0[12];
  float4 color;
  float4 line_info;
};

struct wire_frag_out {
  float4 f_color [[color(0)]];
};

fragment wire_frag_out wire_frag(constant WireUniforms& uniforms
                                 [[buffer(0)]]) {
  wire_frag_out stage_output = {};
  stage_output.f_color = uniforms.color;
  return stage_output;
}

static_assert(sizeof(WireUniforms) == 112, "Uniform size");
static_assert(__builtin_offsetof(WireUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(WireUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(WireUniforms, color) == 80, "color offset");
static_assert(__builtin_offsetof(WireUniforms, line_info) == 96,
              "line_info offset");
