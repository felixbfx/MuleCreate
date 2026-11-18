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

struct PaperUniforms {
  float3 paper;
  float2 size;
  float stain;
  float grain;
  float vignette;
  char padding0[12];
  float3 cam_right;
  float3 cam_up;
  packed_float3 cam_back;
  float view_tan;
  float4 line_info;
};

struct paper_vert_out {
  float4 position [[position]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread uint& vertex_index, thread float4& clip_position) {
  float2 triangle_corner =
      float2(float((int(vertex_index) << 1) & 2), float(int(vertex_index) & 2));
  clip_position = float4((triangle_corner * 2.0) - float2(1.0), 0.999, 1.0);
}

vertex paper_vert_out paper_vert(uint vertex_index [[vertex_id]]) {
  paper_vert_out stage_output = {};
  shade_vertex(vertex_index, stage_output.position);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(PaperUniforms) == 112, "Uniform size");
static_assert(__builtin_offsetof(PaperUniforms, paper) == 0, "paper offset");
static_assert(__builtin_offsetof(PaperUniforms, size) == 16, "size offset");
static_assert(__builtin_offsetof(PaperUniforms, stain) == 24, "stain offset");
static_assert(__builtin_offsetof(PaperUniforms, grain) == 28, "grain offset");
static_assert(__builtin_offsetof(PaperUniforms, vignette) == 32,
              "vignette offset");
static_assert(__builtin_offsetof(PaperUniforms, cam_right) == 48,
              "cam_right offset");
static_assert(__builtin_offsetof(PaperUniforms, cam_up) == 64, "cam_up offset");
static_assert(__builtin_offsetof(PaperUniforms, cam_back) == 80,
              "cam_back offset");
static_assert(__builtin_offsetof(PaperUniforms, view_tan) == 92,
              "view_tan offset");
static_assert(__builtin_offsetof(PaperUniforms, line_info) == 96,
              "line_info offset");
