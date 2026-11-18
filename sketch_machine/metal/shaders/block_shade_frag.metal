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

// Implementation of signed integer mod accurate to SPIR-V specification
template <typename DividendType, typename DivisorType>
inline DividendType signed_modulo(DividendType dividend, DivisorType divisor) {
  DividendType remainder = dividend - divisor * (dividend / divisor);
  return select(DividendType(remainder + divisor), remainder,
                remainder == 0 || (dividend >= 0) == (divisor >= 0));
}

struct BlockShadeUniforms {
  float4x4 mvp;
  int vis_col;
  char padding0[12];
  packed_float3 eye;
  float alpha;
  float lift;
  float light_on;
  char padding1[8];
  float3 sun_dir;
  float3 sun_col;
  packed_float3 sky_col;
  float away_dark;
  float shadow_dark;
  float bands;
  float shadow_on;
  char padding2[4];
  float4x4 shadow_vp;
  float shadow_bias;
  float shadow_lift;
  float dome;
  char padding3[4];
  float4 line_info;
};

struct block_shade_frag_out {
  float4 f_color [[color(0)]];
};

struct block_shade_frag_in {
  float3 v_world [[user(locn0)]];
  float3 v_color [[user(locn1)]];
  int v_slot [[user(locn2)]];
};

static inline __attribute__((always_inline)) float4
tab(thread const int& mesh_index, thread const int& texel_column,
    texture2d<float> mesh_table, sampler mesh_table_sampler) {
  int linear_texel_index = (mesh_index * 3) + texel_column;
  return mesh_table.read(uint2(int2(signed_modulo(linear_texel_index, 1024),
                                    linear_texel_index / 1024)),
                         0);
}

static inline __attribute__((always_inline)) float band(
    thread const float& light_value, constant BlockShadeUniforms& uniforms) {
  if (uniforms.bands < 1.5) {
    return light_value;
  }

  return floor((fast::clamp(light_value, 0.0, 1.0) * (uniforms.bands - 1.0)) +
               0.5) /
         (uniforms.bands - 1.0);
}

static inline __attribute__((always_inline)) float in_shadow(
    thread const float3& world_position, constant BlockShadeUniforms& uniforms,
    texture2d<float> shadow_map, sampler shadow_sampler) {
  if (uniforms.shadow_on < 0.5) {
    return 0.0;
  }

  float4 shadow_clip_position =
      uniforms.shadow_vp * float4(world_position, 1.0);
  float3 shadow_coordinates =
      ((shadow_clip_position.xyz / float3(shadow_clip_position.w)) * 0.5) +
      float3(0.5);
  bool outside_left = shadow_coordinates.x <= 0.0;
  bool outside_horizontal_bounds;

  if (!outside_left) {
    outside_horizontal_bounds = shadow_coordinates.x >= 1.0;
  } else {
    outside_horizontal_bounds = outside_left;
  }

  bool outside_horizontal_or_below;

  if (!outside_horizontal_bounds) {
    outside_horizontal_or_below = shadow_coordinates.y <= 0.0;
  } else {
    outside_horizontal_or_below = outside_horizontal_bounds;
  }

  bool outside_xy_bounds;

  if (!outside_horizontal_or_below) {
    outside_xy_bounds = shadow_coordinates.y >= 1.0;
  } else {
    outside_xy_bounds = outside_horizontal_or_below;
  }

  bool outside_shadow_bounds;

  if (!outside_xy_bounds) {
    outside_shadow_bounds = shadow_coordinates.z >= 1.0;
  } else {
    outside_shadow_bounds = outside_xy_bounds;
  }

  if (outside_shadow_bounds) {
    return 0.0;
  }

  float2 shadow_texel_size =
      float2(1.0) /
      float2(int2(shadow_map.get_width(), shadow_map.get_height()));
  float occluded_samples = 0.0;
  for (int sample_offset_x = -1; sample_offset_x <= 1; sample_offset_x++) {
    for (int sample_offset_y = -1; sample_offset_y <= 1; sample_offset_y++) {
      occluded_samples += float(
          (shadow_coordinates.z - uniforms.shadow_bias) >
          shadow_map
              .sample(shadow_sampler,
                      (shadow_coordinates.xy +
                       (float2(float(sample_offset_x), float(sample_offset_y)) *
                        shadow_texel_size)))
              .x);
    }
  }

  return occluded_samples / 9.0;
}

static inline __attribute__((always_inline)) float3 sun_and_sky(
    thread const float3& surface_normal, thread const float& direct_light,
    thread const float3& sky_color, constant BlockShadeUniforms& uniforms) {
  float hemisphere_factor = 0.5 + (0.5 * surface_normal.y);
  float hemisphere_light = band(hemisphere_factor, uniforms);
  return mix(float3(direct_light),
             mix(float3(1.0), sky_color, float3(0.5)) * hemisphere_light,
             float3(uniforms.dome));
}

fragment block_shade_frag_out
block_shade_frag(block_shade_frag_in stage_input [[stage_in]],
                 constant BlockShadeUniforms& uniforms [[buffer(0)]],
                 texture2d<float> mesh_table [[texture(0)]],
                 texture2d<float> shadow_map [[texture(1)]],
                 sampler mesh_table_sampler [[sampler(0)]],
                 sampler shadow_sampler [[sampler(1)]]) {
  block_shade_frag_out stage_output = {};
  int tint_mesh_index = stage_input.v_slot;
  int tint_column = 0;
  float4 mesh_tint =
      tab(tint_mesh_index, tint_column, mesh_table, mesh_table_sampler);
  float3 surface_normal = fast::normalize(
      cross(dfdx(stage_input.v_world), dfdy(stage_input.v_world)));
  float3 view_direction =
      fast::normalize(float3(uniforms.eye) - stage_input.v_world);

  if (dot(surface_normal, view_direction) < 0.0) {
    surface_normal = -surface_normal;
  }

  float3 shaded_color;

  if (uniforms.light_on > 0.5) {
    float sun_facing = dot(surface_normal, uniforms.sun_dir);
    float sun_illumination_input = smoothstep(-0.25, 0.9, sun_facing);
    float sun_illumination = band(sun_illumination_input, uniforms);
    float3 shadow_offset_normal;

    if (sun_facing >= 0.0) {
      shadow_offset_normal = surface_normal;
    } else {
      shadow_offset_normal = -surface_normal;
    }

    float3 shadow_query_position =
        (stage_input.v_world + (shadow_offset_normal * uniforms.lift)) +
        (uniforms.sun_dir * uniforms.lift);
    float shadow_sample =
        in_shadow(shadow_query_position, uniforms, shadow_map, shadow_sampler);
    float shadow_coverage = band(shadow_sample, uniforms);
    float light_amount =
        fast::min(mix(1.0 - uniforms.away_dark, 1.0, sun_illumination),
                  1.0 - (uniforms.shadow_dark * shadow_coverage));
    float3 lighting_sample_normal = surface_normal;
    float lighting_sample_amount = light_amount;
    float3 sky_sample_color = float3(uniforms.sky_col);
    shaded_color = (stage_input.v_color *
                    sun_and_sky(lighting_sample_normal, lighting_sample_amount,
                                sky_sample_color, uniforms)) *
                   mix(float3(uniforms.sky_col), uniforms.sun_col,
                       float3(sun_illumination * (1.0 - shadow_coverage)));
  } else {
    float view_illumination_input = dot(surface_normal, view_direction);
    float view_illumination = band(view_illumination_input, uniforms);
    shaded_color = stage_input.v_color * (0.72 + (0.28 * view_illumination));
  }

  stage_output.f_color = float4(
      mix(shaded_color, mesh_tint.xyz, float3(mesh_tint.w)), uniforms.alpha);
  return stage_output;
}

static_assert(sizeof(BlockShadeUniforms) == 272, "Uniform size");
static_assert(__builtin_offsetof(BlockShadeUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, vis_col) == 64,
              "vis_col offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, eye) == 80, "eye offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, alpha) == 92,
              "alpha offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, lift) == 96,
              "lift offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, light_on) == 100,
              "light_on offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, sun_dir) == 112,
              "sun_dir offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, sun_col) == 128,
              "sun_col offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, sky_col) == 144,
              "sky_col offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, away_dark) == 156,
              "away_dark offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, shadow_dark) == 160,
              "shadow_dark offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, bands) == 164,
              "bands offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, shadow_on) == 168,
              "shadow_on offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, shadow_vp) == 176,
              "shadow_vp offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, shadow_bias) == 240,
              "shadow_bias offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, shadow_lift) == 244,
              "shadow_lift offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, dome) == 248,
              "dome offset");
static_assert(__builtin_offsetof(BlockShadeUniforms, line_info) == 256,
              "line_info offset");
