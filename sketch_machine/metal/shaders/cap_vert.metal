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

// Returns the inverse of a matrix, by using the algorithm of calculating the
// classical adjoint and dividing by the determinant. The contents of the matrix
// are changed.
static inline __attribute__((always_inline)) float2x2
inverse_matrix2(float2x2 matrix) {
  float2x2
      adjugate;  // The adjoint matrix (inverse after dividing by determinant)

  // Create the transpose of the cofactors, as the classical adjoint of the
  // matrix.
  adjugate[0][0] = matrix[1][1];
  adjugate[0][1] = -matrix[0][1];

  adjugate[1][0] = -matrix[1][0];
  adjugate[1][1] = matrix[0][0];

  // Calculate the determinant as a combination of the cofactors of the first
  // row.
  float matrix_determinant =
      (adjugate[0][0] * matrix[0][0]) + (adjugate[0][1] * matrix[1][0]);

  // Divide the classical adjoint matrix by the determinant.
  // If determinant is zero, matrix is not invertable, so leave it unchanged.
  return (matrix_determinant != 0.0f) ? (adjugate * (1.0f / matrix_determinant))
                                      : matrix;
}

struct CapUniforms {
  float4x4 vp;
  float2 half_vp;
  float focal;
  float min_px;
  float bias0;
  float grain;
  float fade;
  float light_on;
  float3 sun_dir;
  packed_float3 sky_col;
  float shadow_dark;
  float bands;
  float shadow_on;
  char padding0[8];
  float4x4 shadow_vp;
  float shadow_bias;
  float shadow_lift;
  float dome;
  char padding1[4];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float occl_on;
  float xray_behind;
  char padding2[8];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  int plane_n;
  char padding3[8];
  float3 plane_o[16];
  float3 plane_u[16];
  float3 plane_v[16];
  float4 plane_c[16];
  float4 plane_k[16];
  float4 plane_s[16];
  float4 line_info;
};

struct cap_vert_out {
  float4 v_color [[user(locn0)]];
  float v_w [[user(locn1)]];
  float v_seed [[user(locn2)]];
  float v_px [[user(locn3)]];
  float3 v_wpos [[user(locn4)]];
  float4 v_owner [[user(locn5)]];
  float v_brush [[user(locn6)]];
  float v_flat [[user(locn7)]];
  float2 brush_inverse_column0 [[user(locn8)]];
  float2 brush_inverse_column1 [[user(locn9)]];
  float v_size [[user(locn10)]];
  float4 position [[position]];
  float point_size [[point_size]];
};

struct cap_vert_in {
  float3 in_pos [[attribute(0)]];
  float in_width [[attribute(1)]];
  float4 in_color [[attribute(2)]];
  float4 in_owner [[attribute(3)]];
  float in_style [[attribute(4)]];
  float3 in_normal [[attribute(5)]];
  float in_layer [[attribute(6)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& stroke_world_position_output, thread float3& vertex_position,
    thread float4& owner_ids_output, thread float4& vertex_owner_ids,
    thread float& brush_style_output, thread float& brush_style,
    constant CapUniforms& uniforms, thread float& flat_brush_output,
    thread float2x2& inverse_brush_matrix, thread float4& clip_position,
    thread float& point_diameter, thread float& stroke_width,
    thread float3& vertex_normal, thread float& stroke_layer,
    thread float& point_size_output, thread float4& color_output,
    thread float4& vertex_color, thread float& clip_w_output,
    thread float& noise_seed_output, thread float& pixel_width_output) {
  stroke_world_position_output = vertex_position;
  owner_ids_output = vertex_owner_ids;
  brush_style_output = step(1.5, brush_style);
  float4 center_clip_position = uniforms.vp * float4(vertex_position, 1.0);
  float safe_clip_w = fast::max(center_clip_position.w, 1e-5);
  flat_brush_output = 0.0;
  inverse_brush_matrix = float2x2(float2(1.0, 0.0), float2(0.0, 1.0));
  point_diameter = fast::max(
      ((stroke_width * uniforms.focal) * uniforms.half_vp.y) / safe_clip_w,
      uniforms.min_px);
  clip_position =
      float4(center_clip_position.xy,
             center_clip_position.z - (uniforms.bias0 * center_clip_position.w),
             center_clip_position.w);

  if (length(vertex_normal) > 0.5) {
    float3 surface_normal = fast::normalize(vertex_normal);
    float3 surface_axis_u = fast::normalize(cross(
        surface_normal, select(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0),
                               bool3(abs(surface_normal.y) < 0.9))));
    float3 surface_axis_v = cross(surface_normal, surface_axis_u);
    float brush_radius = stroke_width * 0.5;
    float2 center_screen_position =
        (center_clip_position.xy / float2(safe_clip_w)) * uniforms.half_vp;
    float4 axis_u_clip_position =
        uniforms.vp *
        float4(vertex_position + (surface_axis_u * brush_radius), 1.0);
    float4 axis_v_clip_position =
        uniforms.vp *
        float4(vertex_position + (surface_axis_v * brush_radius), 1.0);
    float2 axis_u_screen_offset =
        ((axis_u_clip_position.xy /
          float2(fast::max(axis_u_clip_position.w, 1e-5))) *
         uniforms.half_vp) -
        center_screen_position;
    float2 axis_v_screen_offset =
        ((axis_v_clip_position.xy /
          float2(fast::max(axis_v_clip_position.w, 1e-5))) *
         uniforms.half_vp) -
        center_screen_position;
    float2x2 brush_screen_matrix =
        float2x2(float2(axis_u_screen_offset), float2(axis_v_screen_offset));
    flat_brush_output = float(abs(determinant(brush_screen_matrix)) > 1e-6);
    float2x2 inverse_brush_projection;

    if (flat_brush_output > 0.5) {
      inverse_brush_projection = inverse_matrix2(brush_screen_matrix);
    } else {
      inverse_brush_projection = float2x2(float2(1.0, 0.0), float2(0.0, 1.0));
    }

    inverse_brush_matrix = inverse_brush_projection;
    point_diameter =
        (2.0 * sqrt(dot(axis_u_screen_offset, axis_u_screen_offset) +
                    dot(axis_v_screen_offset, axis_v_screen_offset))) +
        2.0;
    clip_position =
        float4(center_clip_position.xy,
               (0.9 - (1.8 * stroke_layer)) * center_clip_position.w,
               center_clip_position.w);
  }

  point_size_output = point_diameter;
  color_output = vertex_color;
  clip_w_output = center_clip_position.w;
  noise_seed_output =
      fract(dot(vertex_position, float3(12.9898, 78.233, 37.719)));
  pixel_width_output = point_diameter;
}

vertex cap_vert_out cap_vert(cap_vert_in stage_input [[stage_in]],
                             constant CapUniforms& uniforms [[buffer(0)]]) {
  cap_vert_out stage_output = {};
  float2x2 inverse_brush_matrix = {};
  shade_vertex(stage_output.v_wpos, stage_input.in_pos, stage_output.v_owner,
               stage_input.in_owner, stage_output.v_brush, stage_input.in_style,
               uniforms, stage_output.v_flat, inverse_brush_matrix,
               stage_output.position, stage_output.point_size,
               stage_input.in_width, stage_input.in_normal,
               stage_input.in_layer, stage_output.v_size, stage_output.v_color,
               stage_input.in_color, stage_output.v_w, stage_output.v_seed,
               stage_output.v_px);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  stage_output.brush_inverse_column0 = inverse_brush_matrix[0];
  stage_output.brush_inverse_column1 = inverse_brush_matrix[1];
  return stage_output;
}

static_assert(sizeof(CapUniforms) == 1856, "Uniform size");
static_assert(__builtin_offsetof(CapUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(CapUniforms, half_vp) == 64, "half_vp offset");
static_assert(__builtin_offsetof(CapUniforms, focal) == 72, "focal offset");
static_assert(__builtin_offsetof(CapUniforms, min_px) == 76, "min_px offset");
static_assert(__builtin_offsetof(CapUniforms, bias0) == 80, "bias offset");
static_assert(__builtin_offsetof(CapUniforms, grain) == 84, "grain offset");
static_assert(__builtin_offsetof(CapUniforms, fade) == 88, "fade offset");
static_assert(__builtin_offsetof(CapUniforms, light_on) == 92,
              "light_on offset");
static_assert(__builtin_offsetof(CapUniforms, sun_dir) == 96, "sun_dir offset");
static_assert(__builtin_offsetof(CapUniforms, sky_col) == 112,
              "sky_col offset");
static_assert(__builtin_offsetof(CapUniforms, shadow_dark) == 124,
              "shadow_dark offset");
static_assert(__builtin_offsetof(CapUniforms, bands) == 128, "bands offset");
static_assert(__builtin_offsetof(CapUniforms, shadow_on) == 132,
              "shadow_on offset");
static_assert(__builtin_offsetof(CapUniforms, shadow_vp) == 144,
              "shadow_vp offset");
static_assert(__builtin_offsetof(CapUniforms, shadow_bias) == 208,
              "shadow_bias offset");
static_assert(__builtin_offsetof(CapUniforms, shadow_lift) == 212,
              "shadow_lift offset");
static_assert(__builtin_offsetof(CapUniforms, dome) == 216, "dome offset");
static_assert(__builtin_offsetof(CapUniforms, eye) == 224, "eye offset");
static_assert(__builtin_offsetof(CapUniforms, view_back) == 240,
              "view_back offset");
static_assert(__builtin_offsetof(CapUniforms, ortho) == 252, "ortho offset");
static_assert(__builtin_offsetof(CapUniforms, occl_on) == 256,
              "occl_on offset");
static_assert(__builtin_offsetof(CapUniforms, xray_behind) == 260,
              "xray_behind offset");
static_assert(__builtin_offsetof(CapUniforms, occl_tol) == 272,
              "occl_tol offset");
static_assert(__builtin_offsetof(CapUniforms, occl_min) == 284,
              "occl_min offset");
static_assert(__builtin_offsetof(CapUniforms, aid_pass) == 288,
              "aid_pass offset");
static_assert(__builtin_offsetof(CapUniforms, plane_n) == 292,
              "plane_n offset");
static_assert(__builtin_offsetof(CapUniforms, plane_o) == 304,
              "plane_o offset");
static_assert(__builtin_offsetof(CapUniforms, plane_u) == 560,
              "plane_u offset");
static_assert(__builtin_offsetof(CapUniforms, plane_v) == 816,
              "plane_v offset");
static_assert(__builtin_offsetof(CapUniforms, plane_c) == 1072,
              "plane_c offset");
static_assert(__builtin_offsetof(CapUniforms, plane_k) == 1328,
              "plane_k offset");
static_assert(__builtin_offsetof(CapUniforms, plane_s) == 1584,
              "plane_s offset");
static_assert(__builtin_offsetof(CapUniforms, line_info) == 1840,
              "line_info offset");
