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

struct RibbonUniforms {
  float4x4 vp;
  float2 half_vp;
  float focal;
  float min_px;
  float bias0;
  char padding0[12];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  packed_float3 sun_dir;
  float grain;
  float fade;
  float light_on;
  char padding1[8];
  float3 sun_col;
  packed_float3 sky_col;
  float shadow_dark;
  float bands;
  float shadow_on;
  char padding2[8];
  float4x4 shadow_vp;
  float shadow_bias;
  float shadow_lift;
  float dome;
  float occl_on;
  float xray_behind;
  char padding3[12];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  int plane_n;
  char padding4[8];
  float3 plane_o[16];
  float3 plane_u[16];
  float3 plane_v[16];
  float4 plane_c[16];
  float4 plane_k[16];
  float4 plane_s[16];
  float edge_gain;
  char padding5[12];
  float4 line_info;
};

struct ribbon_vert_out {
  float4 v_color [[user(locn0)]];
  float v_w [[user(locn1)]];
  float v_side [[user(locn2)]];
  float v_arc [[user(locn3)]];
  float v_px [[user(locn4)]];
  float v_brush [[user(locn5)]];
  float v_light [[user(locn6)]];
  float3 v_wpos [[user(locn7)]];
  float3 v_tan [[user(locn8)]];
  float v_rad [[user(locn9)]];
  float4 v_owner [[user(locn10)]];
  float4 position [[position]];
};

struct ribbon_vert_in {
  float3 in_pos [[attribute(0)]];
  float3 in_prev [[attribute(1)]];
  float3 in_next [[attribute(2)]];
  float in_width [[attribute(3)]];
  float in_side [[attribute(4)]];
  float in_arc [[attribute(5)]];
  float4 in_color [[attribute(6)]];
  float4 in_owner [[attribute(7)]];
  float3 in_normal [[attribute(8)]];
  float in_layer [[attribute(9)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& stroke_world_position_output, thread float3& vertex_position,
    thread float4& owner_ids_output, thread float4& vertex_owner_ids,
    thread float& stroke_side, thread float3& vertex_normal,
    thread float3& next_vertex_position,
    thread float3& previous_vertex_position, constant RibbonUniforms& uniforms,
    thread float& stroke_width, thread float4& clip_position,
    thread float& stroke_layer, thread float4& color_output,
    thread float4& vertex_color, thread float& clip_w_output,
    thread float& stroke_side_output, thread float& arc_length_output,
    thread float& stroke_arc_length, thread float& pixel_width_output,
    thread float& brush_style_output, thread float& edge_light_output,
    thread float3& tangent_output, thread float& radius_output) {
  stroke_world_position_output = vertex_position;
  owner_ids_output = vertex_owner_ids;
  float brush_style_flag = step(1.5, abs(stroke_side));

  if (length(vertex_normal) > 0.5) {
    float flat_stroke_side = stroke_side / (1.0 + brush_style_flag);
    float3 surface_tangent = next_vertex_position - previous_vertex_position;
    float3 normalized_surface_tangent;

    if (length(surface_tangent) > 1e-9) {
      normalized_surface_tangent = fast::normalize(surface_tangent);
    } else {
      normalized_surface_tangent = float3(1.0, 0.0, 0.0);
    }

    surface_tangent = normalized_surface_tangent;
    float3 surface_bitangent =
        cross(fast::normalize(vertex_normal), surface_tangent);
    float3 normalized_surface_bitangent;

    if (length(surface_bitangent) > 1e-6) {
      normalized_surface_bitangent = fast::normalize(surface_bitangent);
    } else {
      normalized_surface_bitangent = float3(0.0, 1.0, 0.0);
    }

    surface_bitangent = normalized_surface_bitangent;
    float4 flat_clip_position =
        uniforms.vp *
        float4(
            vertex_position +
                (((surface_bitangent * flat_stroke_side) * stroke_width) * 0.5),
            1.0);
    float flat_clip_w = fast::max(flat_clip_position.w, 1e-5);
    clip_position = float4(flat_clip_position.xy,
                           (0.9 - (1.8 * stroke_layer)) * flat_clip_position.w,
                           flat_clip_position.w);
    color_output = vertex_color;
    clip_w_output = flat_clip_position.w;
    stroke_side_output = flat_stroke_side;
    arc_length_output = stroke_arc_length;
    pixel_width_output = fast::max(
        ((stroke_width * uniforms.focal) * uniforms.half_vp.y) / flat_clip_w,
        uniforms.min_px);
    brush_style_output = brush_style_flag;
    edge_light_output = 0.0;
    tangent_output = float3(0.0);
    radius_output = 0.0;
    return;
  }

  float screen_stroke_side = stroke_side / (1.0 + brush_style_flag);
  float4 center_clip_position = uniforms.vp * float4(vertex_position, 1.0);
  float4 previous_clip_position =
      uniforms.vp * float4(previous_vertex_position, 1.0);
  float4 next_clip_position = uniforms.vp * float4(next_vertex_position, 1.0);
  float safe_clip_w = fast::max(center_clip_position.w, 1e-5);
  float2 center_screen_position =
      (center_clip_position.xy / float2(safe_clip_w)) * uniforms.half_vp;
  float2 screen_direction =
      ((next_clip_position.xy / float2(fast::max(next_clip_position.w, 1e-5))) *
       uniforms.half_vp) -
      ((previous_clip_position.xy /
        float2(fast::max(previous_clip_position.w, 1e-5))) *
       uniforms.half_vp);
  if (length(screen_direction) < 1e-6) {
    screen_direction = float2(1.0, 0.0);
  }

  float2 screen_normal =
      fast::normalize(float2(-screen_direction.y, screen_direction.x));
  float pixel_width = fast::max(
      ((stroke_width * uniforms.focal) * uniforms.half_vp.y) / safe_clip_w,
      uniforms.min_px);
  float2 extruded_pixel =
      center_screen_position +
      (((screen_normal * pixel_width) * 0.5) * screen_stroke_side);
  clip_position = float4(
      (extruded_pixel / uniforms.half_vp) * safe_clip_w,
      center_clip_position.z - (uniforms.bias0 * safe_clip_w), safe_clip_w);
  color_output = vertex_color;
  clip_w_output = center_clip_position.w;
  stroke_side_output = screen_stroke_side;
  arc_length_output = stroke_arc_length;
  pixel_width_output = pixel_width;
  brush_style_output = brush_style_flag;
  float3 stroke_tangent = next_vertex_position - previous_vertex_position;
  float3 normalized_stroke_tangent;

  if (length(stroke_tangent) > 1e-9) {
    normalized_stroke_tangent = fast::normalize(stroke_tangent);
  } else {
    normalized_stroke_tangent = float3(1.0, 0.0, 0.0);
  }

  stroke_tangent = normalized_stroke_tangent;
  tangent_output = stroke_tangent;
  radius_output = stroke_width * 0.5;
  float3 selected_view_direction;

  if (uniforms.ortho > 0.5) {
    selected_view_direction = float3(uniforms.view_back);
  } else {
    selected_view_direction = fast::normalize(uniforms.eye - vertex_position);
  }

  float3 view_direction = selected_view_direction;
  float3 view_bitangent = cross(view_direction, stroke_tangent);
  float view_bitangent_length = length(view_bitangent);
  edge_light_output = 0.0;

  if (view_bitangent_length > 1e-5) {
    view_bitangent /= float3(view_bitangent_length);
    float4 bitangent_clip_position =
        uniforms.vp *
        float4(vertex_position +
                   (view_bitangent * fast::max(stroke_width, 0.0001)),
               1.0);
    float2 bitangent_screen_offset =
        ((bitangent_clip_position.xy /
          float2(fast::max(bitangent_clip_position.w, 1e-5))) *
         uniforms.half_vp) -
        center_screen_position;
    if (dot(bitangent_screen_offset, screen_normal) < 0.0) {
      view_bitangent = -view_bitangent;
    }

    edge_light_output = dot(view_bitangent, float3(uniforms.sun_dir));
  }
}

vertex ribbon_vert_out ribbon_vert(ribbon_vert_in stage_input [[stage_in]],
                                   constant RibbonUniforms& uniforms
                                   [[buffer(0)]]) {
  ribbon_vert_out stage_output = {};
  shade_vertex(stage_output.v_wpos, stage_input.in_pos, stage_output.v_owner,
               stage_input.in_owner, stage_input.in_side, stage_input.in_normal,
               stage_input.in_next, stage_input.in_prev, uniforms,
               stage_input.in_width, stage_output.position,
               stage_input.in_layer, stage_output.v_color, stage_input.in_color,
               stage_output.v_w, stage_output.v_side, stage_output.v_arc,
               stage_input.in_arc, stage_output.v_px, stage_output.v_brush,
               stage_output.v_light, stage_output.v_tan, stage_output.v_rad);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(RibbonUniforms) == 1904, "Uniform size");
static_assert(__builtin_offsetof(RibbonUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(RibbonUniforms, half_vp) == 64,
              "half_vp offset");
static_assert(__builtin_offsetof(RibbonUniforms, focal) == 72, "focal offset");
static_assert(__builtin_offsetof(RibbonUniforms, min_px) == 76,
              "min_px offset");
static_assert(__builtin_offsetof(RibbonUniforms, bias0) == 80, "bias offset");
static_assert(__builtin_offsetof(RibbonUniforms, eye) == 96, "eye offset");
static_assert(__builtin_offsetof(RibbonUniforms, view_back) == 112,
              "view_back offset");
static_assert(__builtin_offsetof(RibbonUniforms, ortho) == 124, "ortho offset");
static_assert(__builtin_offsetof(RibbonUniforms, sun_dir) == 128,
              "sun_dir offset");
static_assert(__builtin_offsetof(RibbonUniforms, grain) == 140, "grain offset");
static_assert(__builtin_offsetof(RibbonUniforms, fade) == 144, "fade offset");
static_assert(__builtin_offsetof(RibbonUniforms, light_on) == 148,
              "light_on offset");
static_assert(__builtin_offsetof(RibbonUniforms, sun_col) == 160,
              "sun_col offset");
static_assert(__builtin_offsetof(RibbonUniforms, sky_col) == 176,
              "sky_col offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_dark) == 188,
              "shadow_dark offset");
static_assert(__builtin_offsetof(RibbonUniforms, bands) == 192, "bands offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_on) == 196,
              "shadow_on offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_vp) == 208,
              "shadow_vp offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_bias) == 272,
              "shadow_bias offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_lift) == 276,
              "shadow_lift offset");
static_assert(__builtin_offsetof(RibbonUniforms, dome) == 280, "dome offset");
static_assert(__builtin_offsetof(RibbonUniforms, occl_on) == 284,
              "occl_on offset");
static_assert(__builtin_offsetof(RibbonUniforms, xray_behind) == 288,
              "xray_behind offset");
static_assert(__builtin_offsetof(RibbonUniforms, occl_tol) == 304,
              "occl_tol offset");
static_assert(__builtin_offsetof(RibbonUniforms, occl_min) == 316,
              "occl_min offset");
static_assert(__builtin_offsetof(RibbonUniforms, aid_pass) == 320,
              "aid_pass offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_n) == 324,
              "plane_n offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_o) == 336,
              "plane_o offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_u) == 592,
              "plane_u offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_v) == 848,
              "plane_v offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_c) == 1104,
              "plane_c offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_k) == 1360,
              "plane_k offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_s) == 1616,
              "plane_s offset");
static_assert(__builtin_offsetof(RibbonUniforms, edge_gain) == 1872,
              "edge_gain offset");
static_assert(__builtin_offsetof(RibbonUniforms, line_info) == 1888,
              "line_info offset");
