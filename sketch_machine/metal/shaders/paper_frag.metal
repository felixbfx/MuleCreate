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

struct paper_frag_out {
  float4 f_color [[color(0)]];
};

static inline __attribute__((always_inline)) float hash13(
    thread float3& hash_state) {
  hash_state = fract(hash_state * 0.1031);
  hash_state += float3(dot(hash_state, hash_state.zyx + float3(31.32)));
  return fract((hash_state.x + hash_state.y) * hash_state.z);
}

static inline __attribute__((always_inline)) float vnoise3(
    thread const float3& noise_coordinates) {
  float3 cell_origin = floor(noise_coordinates);
  float3 blend_weights = fract(noise_coordinates);
  blend_weights =
      (blend_weights * blend_weights) * (float3(3.0) - (blend_weights * 2.0));
  float3 corner000 = cell_origin;
  float noise000 = hash13(corner000);
  float3 corner100 = cell_origin + float3(1.0, 0.0, 0.0);
  float noise100 = hash13(corner100);
  float3 corner010 = cell_origin + float3(0.0, 1.0, 0.0);
  float noise010 = hash13(corner010);
  float3 corner110 = cell_origin + float3(1.0, 1.0, 0.0);
  float noise110 = hash13(corner110);
  float lower_plane_noise =
      mix(mix(noise000, noise100, blend_weights.x),
          mix(noise010, noise110, blend_weights.x), blend_weights.y);
  float3 corner001 = cell_origin + float3(0.0, 0.0, 1.0);
  float noise001 = hash13(corner001);
  float3 corner101 = cell_origin + float3(1.0, 0.0, 1.0);
  float noise101 = hash13(corner101);
  float3 corner011 = cell_origin + float3(0.0, 1.0, 1.0);
  float noise011 = hash13(corner011);
  float3 corner111 = cell_origin + float3(1.0);
  float noise111 = hash13(corner111);
  float upper_plane_noise =
      mix(mix(noise001, noise101, blend_weights.x),
          mix(noise011, noise111, blend_weights.x), blend_weights.y);
  return mix(lower_plane_noise, upper_plane_noise, blend_weights.z);
}

fragment paper_frag_out paper_frag(constant PaperUniforms& uniforms
                                   [[buffer(0)]],
                                   float4 fragment_position [[position]]) {
  paper_frag_out stage_output = {};
  float2 screen_coordinates =
      ((fragment_position.xy / uniforms.size) * 2.0) - float2(1.0);
  float3 view_direction = fast::normalize(
      ((-float3(uniforms.cam_back)) +
       (uniforms.cam_right * ((screen_coordinates.x * uniforms.view_tan) *
                              (uniforms.size.x / uniforms.size.y)))) +
      (uniforms.cam_up * (screen_coordinates.y * uniforms.view_tan)));
  float3 coarse_blotch_coordinates = view_direction * 3.0;
  float3 fine_blotch_coordinates = (view_direction * 7.0) + float3(5.0);
  float blotch_noise = ((vnoise3(coarse_blotch_coordinates) * 0.6) +
                        (vnoise3(fine_blotch_coordinates) * 0.4)) -
                       0.5;
  float3 horizontal_fiber_coordinates =
      view_direction * float3(380.0, 70.0, 380.0);
  float3 vertical_fiber_coordinates =
      (view_direction * float3(70.0, 380.0, 70.0)) + float3(9.0);
  float fiber_noise = ((vnoise3(horizontal_fiber_coordinates) +
                        vnoise3(vertical_fiber_coordinates)) *
                       0.5) -
                      0.5;
  float3 tooth_coordinates = view_direction * 1300.0;
  float tooth_noise = vnoise3(tooth_coordinates) - 0.5;
  float2 vignette_coordinates =
      (fragment_position.xy / uniforms.size) - float2(0.5);
  float paper_shade =
      (1.0 + ((uniforms.stain * 2.0) * blotch_noise)) +
      (uniforms.grain * ((0.9 * tooth_noise) + (0.6 * fiber_noise)));
  paper_shade *=
      (1.0 -
       ((uniforms.vignette * dot(vignette_coordinates, vignette_coordinates)) *
        1.6));
  stage_output.f_color = float4(uniforms.paper * paper_shade, 1.0);
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
