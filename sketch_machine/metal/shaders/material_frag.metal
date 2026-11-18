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
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <simd/simd.h>

#include <metal_stdlib>

using namespace metal;

template <typename T, size_t ElementCount>
struct ShaderArray {
  T elements[ElementCount ? ElementCount : 1];

  thread T& operator[](size_t element_index) thread {
    return elements[element_index];
  }
  constexpr const thread T& operator[](size_t element_index) const thread {
    return elements[element_index];
  }

  device T& operator[](size_t element_index) device {
    return elements[element_index];
  }
  constexpr const device T& operator[](size_t element_index) const device {
    return elements[element_index];
  }

  constexpr const constant T& operator[](size_t element_index) const constant {
    return elements[element_index];
  }

  threadgroup T& operator[](size_t element_index) threadgroup {
    return elements[element_index];
  }
  constexpr const threadgroup T& operator[](size_t element_index) const
      threadgroup {
    return elements[element_index];
  }
};

// Implementation of the GLSL mod() function, which is slightly different than
// Metal fmod()
template <typename DividendType, typename DivisorType>
inline DividendType mod(DividendType dividend, DivisorType divisor) {
  return dividend - divisor * floor(dividend / divisor);
}

// Implementation of signed integer mod accurate to SPIR-V specification
template <typename DividendType, typename DivisorType>
inline DividendType signed_modulo(DividendType dividend, DivisorType divisor) {
  DividendType remainder = dividend - divisor * (dividend / divisor);
  return select(DividendType(remainder + divisor), remainder,
                remainder == 0 || (dividend >= 0) == (divisor >= 0));
}

struct MaterialUniforms {
  float4x4 vp;
  float bias0;
  int occluders_only;
  int surf_rows;
  float lift_min;
  float stain;
  float layer;
  float front_alpha;
  float mono;
  float light_on;
  char padding0[12];
  float3 sun_dir;
  float3 sun_col;
  packed_float3 sky_col;
  float away_dark;
  float shadow_dark;
  float calm;
  float bands;
  float shadow_on;
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
  char padding3[12];
  float4 line_info;
};

constant ShaderArray<float, 5> material_texture_scales =
    ShaderArray<float, 5>({1.0, 0.45, 0.4, 0.5, 0.35});
constant ShaderArray<float, 5> material_brightness_gains =
    ShaderArray<float, 5>({1.0, 0.9325, 0.9664, 0.957, 0.976});
constant ShaderArray<float, 5> material_stain_weights =
    ShaderArray<float, 5>({1.0, 0.0, 1.0, 0.0, 1.0});

struct material_frag_out {
  float4 f_color [[color(0)]];
};

struct material_frag_in {
  float3 v_world [[user(locn0)]];
  float3 v_normal [[user(locn1)]];
  float3 v_calm [[user(locn2)]];
  int v_slot [[user(locn3)]];
};

static inline __attribute__((always_inline)) float4
surf(thread const int& surface_index, thread const int& texel_column,
     texture2d<float> surface_table, sampler surface_table_sampler,
     constant MaterialUniforms& uniforms) {
  return surface_table.read(
      uint2(int2(texel_column + (10 * (surface_index / uniforms.surf_rows)),
                 signed_modulo(surface_index, uniforms.surf_rows))),
      0);
}

static inline __attribute__((always_inline)) void load_surface(
    texture2d<float> surface_table, sampler surface_table_sampler,
    constant MaterialUniforms& uniforms, thread int& surface_index,
    thread float3& surface_tint, thread int& material_index,
    thread float3& texture_anchor, thread float& texture_scale,
    thread float3& texture_offset, thread float& grid_spacing,
    thread float4& selection_state, thread float3& grid_color,
    thread float& grid_opacity, thread float& shadow_lift,
    thread float& surface_opacity, thread float4& owner_ids) {
  int tint_surface_index = surface_index;
  int tint_column = 0;
  float4 tint_material_texel =
      surf(tint_surface_index, tint_column, surface_table,
           surface_table_sampler, uniforms);
  int anchor_surface_index = surface_index;
  int anchor_column = 1;
  float4 anchor_scale_texel =
      surf(anchor_surface_index, anchor_column, surface_table,
           surface_table_sampler, uniforms);
  int offset_surface_index = surface_index;
  int offset_column = 2;
  float4 offset_grid_texel =
      surf(offset_surface_index, offset_column, surface_table,
           surface_table_sampler, uniforms);
  int grid_surface_index = surface_index;
  int grid_column = 4;
  float4 grid_color_texel = surf(grid_surface_index, grid_column, surface_table,
                                 surface_table_sampler, uniforms);
  int shadow_surface_index = surface_index;
  int shadow_column = 5;
  float4 shadow_opacity_owner_texel =
      surf(shadow_surface_index, shadow_column, surface_table,
           surface_table_sampler, uniforms);
  surface_tint = tint_material_texel.xyz;
  material_index = int(tint_material_texel.w + 0.5);
  texture_anchor = anchor_scale_texel.xyz;
  texture_scale = anchor_scale_texel.w;
  texture_offset = offset_grid_texel.xyz;
  grid_spacing = offset_grid_texel.w;
  int state_surface_index = surface_index;
  int state_column = 3;
  selection_state = surf(state_surface_index, state_column, surface_table,
                         surface_table_sampler, uniforms);
  grid_color = grid_color_texel.xyz;
  grid_opacity = grid_color_texel.w;
  shadow_lift = fast::max(shadow_opacity_owner_texel.x, uniforms.lift_min);
  surface_opacity = shadow_opacity_owner_texel.y;
  owner_ids = float4(shadow_opacity_owner_texel.z, -1.0, -1.0, -1.0);
}

static inline __attribute__((always_inline)) bool occl_dense_at(
    thread const int2& pixel_coordinates, constant MaterialUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler) {
  bool density_threshold_disabled = uniforms.occl_min <= 0.0;
  bool is_dense;

  if (!density_threshold_disabled) {
    is_dense =
        (1.0 -
         exp(-occlusion_accumulation.read(uint2(pixel_coordinates), 0).x)) >=
        uniforms.occl_min;
  } else {
    is_dense = density_threshold_disabled;
  }

  return is_dense;
}

static inline __attribute__((always_inline)) float view_dist(
    thread const float3& world_position, constant MaterialUniforms& uniforms) {
  float distance_to_eye;

  if (uniforms.ortho > 0.5) {
    distance_to_eye =
        dot(uniforms.eye - world_position, float3(uniforms.view_back));
  } else {
    distance_to_eye = length(world_position - uniforms.eye);
  }

  return distance_to_eye;
}

static inline __attribute__((always_inline)) float cover_at(
    thread const float3& world_position, thread const float4& owner_ids,
    thread const int2& sample_pixel, constant MaterialUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, texture2d<float> occlusion_map,
    sampler occlusion_sampler) {
  float4 front_occluder = occlusion_map.read(uint2(sample_pixel), 0);
  bool front_sample_empty = front_occluder.z <= 0.0;
  bool front_sample_transparent;

  if (!front_sample_empty) {
    int2 density_pixel = sample_pixel;
    front_sample_transparent =
        !occl_dense_at(density_pixel, uniforms, occlusion_accumulation,
                       occlusion_accumulation_sampler);
  } else {
    front_sample_transparent = front_sample_empty;
  }

  if (front_sample_transparent) {
    return 1.0;
  }

  float3 query_position = world_position;
  float view_distance = view_dist(query_position, uniforms);
  float depth_difference = view_distance - front_occluder.x;
  float depth_epsilon = 2e-5 * abs(view_distance);

  if (depth_difference <= depth_epsilon) {
    return 1.0;
  }

  bool front_owner_valid = front_occluder.w >= 0.0;
  bool shares_owner_id;

  if (front_owner_valid) {
    shares_owner_id = any(owner_ids == float4(front_occluder.w));
  } else {
    shares_owner_id = front_owner_valid;
  }

  bool same_owner = shares_owner_id;
  bool known_front_owner = front_occluder.w >= 0.0;
  bool both_owners_known;

  if (known_front_owner) {
    both_owners_known = fast::max(fast::max(owner_ids.x, owner_ids.y),
                                  fast::max(owner_ids.z, owner_ids.w)) >= 0.0;
  } else {
    both_owners_known = known_front_owner;
  }

  bool owners_known = both_owners_known;
  float owner_tolerance;

  if (same_owner) {
    owner_tolerance = uniforms.occl_tol[0u];
  } else {
    float foreign_tolerance;

    if (owners_known) {
      foreign_tolerance = uniforms.occl_tol[2u];
    } else {
      foreign_tolerance = uniforms.occl_tol[1u];
    }

    owner_tolerance = foreign_tolerance;
  }

  float depth_tolerance = owner_tolerance;
  bool within_projected_tolerance =
      (depth_difference * front_occluder.y) <=
      ((depth_tolerance * front_occluder.z) + depth_epsilon);
  bool within_depth_tolerance;

  if (within_projected_tolerance) {
    within_depth_tolerance =
        depth_difference <= ((same_owner ? 4.0 : 2.0) * front_occluder.z);
  } else {
    within_depth_tolerance = within_projected_tolerance;
  }

  if (within_depth_tolerance) {
    return 1.0;
  }

  return fast::max(uniforms.xray_behind,
                   exp(-occlusion_accumulation.read(uint2(sample_pixel), 0).x));
}

static inline __attribute__((always_inline)) float cover_edge(
    thread const float3& world_position, thread const float4& owner_ids,
    constant MaterialUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, texture2d<float> occlusion_map,
    sampler occlusion_sampler, thread float4& fragment_position) {
  if (uniforms.occl_on < 0.5) {
    return 1.0;
  }

  int2 fragment_pixel = int2(fragment_position.xy);
  int2 occlusion_size =
      int2(occlusion_map.get_width(), occlusion_map.get_height());
  float visibility = 0.0;
  for (int sample_offset_x = -1; sample_offset_x <= 1; sample_offset_x++) {
    for (int sample_offset_y = -1; sample_offset_y <= 1; sample_offset_y++) {
      float3 query_position = world_position;
      float4 query_owner_ids = owner_ids;
      int2 sample_pixel =
          clamp(fragment_pixel + int2(sample_offset_x, sample_offset_y),
                int2(0), occlusion_size - int2(1));
      visibility = fast::max(
          visibility,
          cover_at(query_position, query_owner_ids, sample_pixel, uniforms,
                   occlusion_accumulation, occlusion_accumulation_sampler,
                   occlusion_map, occlusion_sampler));
    }
  }

  return visibility;
}

static inline __attribute__((always_inline)) float fade_px(
    thread const float& feature_size, thread const float& pixel_footprint) {
  return 1.0 -
         smoothstep(0.35, 1.0, pixel_footprint / fast::max(feature_size, 1e-6));
}

static inline __attribute__((always_inline)) float hash12(
    thread const float2& noise_coordinates) {
  float3 hash_state = fract(noise_coordinates.xyx * 0.1031);
  hash_state += float3(dot(hash_state, hash_state.yzx + float3(33.33)));
  return fract((hash_state.x + hash_state.y) * hash_state.z);
}

static inline __attribute__((always_inline)) float vnoise(
    thread const float2& noise_coordinates) {
  float2 cell_origin = floor(noise_coordinates);
  float2 blend_weights = fract(noise_coordinates);
  blend_weights =
      (blend_weights * blend_weights) * (float2(3.0) - (blend_weights * 2.0));
  float2 corner00 = cell_origin;
  float2 corner10 = cell_origin + float2(1.0, 0.0);
  float2 corner01 = cell_origin + float2(0.0, 1.0);
  float2 corner11 = cell_origin + float2(1.0);
  return mix(mix(hash12(corner00), hash12(corner10), blend_weights.x),
             mix(hash12(corner01), hash12(corner11), blend_weights.x),
             blend_weights.y);
}

static inline __attribute__((always_inline)) float fbm2(
    thread const float2& noise_coordinates) {
  float2 octave0_coordinates = noise_coordinates;
  float2 octave1_coordinates = (noise_coordinates * 2.03) + float2(7.1);
  float2 octave2_coordinates = (noise_coordinates * 4.01) + float2(3.7);
  float2 octave3_coordinates = (noise_coordinates * 8.07) + float2(1.3);
  return (((vnoise(octave0_coordinates) * 0.5) +
           (vnoise(octave1_coordinates) * 0.25)) +
          (vnoise(octave2_coordinates) * 0.125)) +
         (vnoise(octave3_coordinates) * 0.0625);
}

static inline __attribute__((always_inline)) float3
m_stoff(thread const float2& texture_coordinates,
        thread const float& pixel_footprint) {
  float weave_frequency = 45.0;
  float2 weave_coordinates = texture_coordinates * weave_frequency;
  float weave_feature_size = 1.0 / weave_frequency;
  float weave_pixel_footprint = pixel_footprint;
  float weave_visibility = fade_px(weave_feature_size, weave_pixel_footprint);
  float2 weave_cell = floor(weave_coordinates);
  float2 cell_fraction = fract(weave_coordinates);
  float2 warp_thickness_coordinates =
      float2(weave_cell.x * 2.3, texture_coordinates.y * 2.5);
  float warp_thickness = 0.75 + (0.5 * vnoise(warp_thickness_coordinates));
  float2 weft_thickness_coordinates =
      float2(texture_coordinates.x * 2.5, (weave_cell.y * 1.9) + 50.0);
  float weft_thickness = 0.75 + (0.5 * vnoise(weft_thickness_coordinates));
  float warp_profile =
      1.0 - powr(abs(cell_fraction.x - 0.5) * 2.0, 2.0 / warp_thickness);
  float weft_profile =
      1.0 - powr(abs(cell_fraction.y - 0.5) * 2.0, 2.0 / weft_thickness);
  float over_under_pattern = mod(weave_cell.x + weave_cell.y, 2.0);
  float thread_profile =
      mix(warp_profile * (0.85 + (0.15 * weft_profile)),
          weft_profile * (0.85 + (0.15 * warp_profile)), over_under_pattern);
  float2 warp_slub_coordinates =
      float2(weave_cell.x * 0.7, texture_coordinates.y * 0.8);
  float2 weft_slub_coordinates =
      float2(texture_coordinates.x * 0.8, (weave_cell.y * 0.7) + 9.0);
  float thread_irregularity =
      ((vnoise(warp_slub_coordinates) + vnoise(weft_slub_coordinates)) * 0.5) -
      0.5;
  float2 cloth_cloud_coordinates = texture_coordinates * 1.5;
  float cloth_tone =
      ((0.968 - (((1.0 - thread_profile) * 0.06) * weave_visibility)) +
       ((thread_irregularity * 0.03) * weave_visibility)) +
      ((fbm2(cloth_cloud_coordinates) - 0.5) * 0.015);
  return float3(cloth_tone, cloth_tone, cloth_tone * 0.994);
}

static inline __attribute__((always_inline)) float3
m_pappe(thread const float2& texture_coordinates,
        thread const float& pixel_footprint) {
  float2 fiber_direction_coordinates = texture_coordinates * 6.0;
  float fiber_angle = fbm2(fiber_direction_coordinates) * 6.2832;
  float2 fiber_direction = float2(cos(fiber_angle), sin(fiber_angle));
  float2 fiber_coordinates = float2(
      dot(texture_coordinates, fiber_direction),
      dot(texture_coordinates, float2(-fiber_direction.y, fiber_direction.x)));
  float2 fiber_noise_coordinates = fiber_coordinates * float2(150.0, 22.0);
  float fiber_feature_size = 0.006666667;
  float fiber_pixel_footprint = pixel_footprint;
  float fiber_noise = (vnoise(fiber_noise_coordinates) - 0.5) *
                      fade_px(fiber_feature_size, fiber_pixel_footprint);
  float2 fine_noise_coordinates = texture_coordinates * 220.0;
  float fine_feature_size = 0.0045454544;
  float fine_pixel_footprint = pixel_footprint;
  float fine_fiber_noise = (vnoise(fine_noise_coordinates) - 0.5) *
                           fade_px(fine_feature_size, fine_pixel_footprint);
  float2 cloud_coordinates = texture_coordinates * 2.5;
  float cloud_noise = fbm2(cloud_coordinates) - 0.5;
  float2 speck_cell = floor(texture_coordinates * 30.0);
  float2 speck_jitter_x_coordinates = speck_cell + float2(3.1);
  float2 speck_jitter_y_coordinates = speck_cell + float2(7.7);
  float2 speck_jitter = float2(hash12(speck_jitter_x_coordinates),
                               hash12(speck_jitter_y_coordinates)) -
                        float2(0.5);
  float2 speck_noise_coordinates = speck_cell;
  float dark_speck =
      step(0.993, hash12(speck_noise_coordinates)) *
      (1.0 -
       smoothstep(0.05, 0.16,
                  length((fract(texture_coordinates * 30.0) - float2(0.5)) -
                         (speck_jitter * 0.6))));
  float speck_feature_size = 0.033333335;
  float speck_pixel_footprint = pixel_footprint;
  dark_speck *= fade_px(speck_feature_size, speck_pixel_footprint);
  float2 light_fiber_coordinates =
      (fiber_coordinates.yx * float2(120.0, 16.0)) + float2(3.0);
  float light_fiber_feature_size = 0.008333334;
  float light_fiber_pixel_footprint = pixel_footprint;
  float light_fibers =
      smoothstep(0.7, 0.9, vnoise(light_fiber_coordinates)) *
      fade_px(light_fiber_feature_size, light_fiber_pixel_footprint);
  float cardboard_tone =
      ((((1.0 + (fiber_noise * 0.13)) + (fine_fiber_noise * 0.05)) +
        (light_fibers * 0.035)) +
       (cloud_noise * 0.05)) -
      (dark_speck * 0.3);
  return float3(0.94, 0.905, 0.85) * cardboard_tone;
}

static inline __attribute__((always_inline)) float3
m_holz(thread const float2& texture_coordinates,
       thread const float& pixel_footprint) {
  float2 coarse_warp_coordinates = texture_coordinates * float2(0.35, 0.08);
  float2 fine_warp_coordinates = texture_coordinates * float2(1.5, 0.3);
  float grain_warp = (fbm2(coarse_warp_coordinates) * 2.5) +
                     (fbm2(fine_warp_coordinates) * 0.35);
  float grain_coordinate = (texture_coordinates.x * 16.0) + (grain_warp * 4.0);
  float grain_line =
      1.0 -
      smoothstep(0.0, 0.09, (abs(fract(grain_coordinate) - 0.5) * 2.0) - 0.0);
  float2 grain_variation_coordinates =
      float2(floor(grain_coordinate) * 3.1, texture_coordinates.y * 1.5);
  grain_line *= (0.5 + (0.5 * vnoise(grain_variation_coordinates)));
  float2 pore_coordinates = texture_coordinates * float2(160.0, 7.0);
  float pore_feature_size = 0.00625;
  float pore_pixel_footprint = pixel_footprint;
  float pore_noise = (vnoise(pore_coordinates) - 0.5) *
                     fade_px(pore_feature_size, pore_pixel_footprint);
  float2 band_coordinates =
      float2(grain_coordinate * 0.15, texture_coordinates.y * 0.1);
  float wood_band_noise = fbm2(band_coordinates) - 0.5;
  float grain_feature_size = 0.0625;
  float grain_pixel_footprint = pixel_footprint;
  float wood_tone =
      ((1.0 - ((grain_line * 0.07) *
               fade_px(grain_feature_size, grain_pixel_footprint))) +
       (pore_noise * 0.04)) +
      (wood_band_noise * 0.04);
  return float3(0.955, 0.915, 0.855) * wood_tone;
}

static inline __attribute__((always_inline)) float3
m_spruehfarbe(thread const float2& texture_coordinates,
              thread const float& pixel_footprint) {
  float2 warp_x_coordinates = texture_coordinates * 3.1;
  float2 warp_y_coordinates = (texture_coordinates * 3.1) + float2(17.3);
  float2 spray_warp =
      float2(fbm2(warp_x_coordinates), fbm2(warp_y_coordinates)) - float2(0.5);
  float2 warped_coordinates = texture_coordinates + (spray_warp * 0.2);
  float transition_width = fast::clamp(pixel_footprint * 6.0, 0.004, 0.08);
  float2 island_coordinates = warped_coordinates * 1.8;
  float paint_islands = smoothstep(
      0.5 - transition_width, 0.5 + transition_width, fbm2(island_coordinates));
  float2 patch_coordinates = (warped_coordinates * 6.5) + float2(5.0);
  float paint_patches =
      smoothstep(0.56 - transition_width, 0.56 + transition_width,
                 fbm2(patch_coordinates));
  float speck_feature_size = 0.020833334;
  float speck_pixel_footprint = pixel_footprint;
  float speck_visibility = fade_px(speck_feature_size, speck_pixel_footprint);
  float fine_feature_size = 0.009090909;
  float fine_pixel_footprint = pixel_footprint;
  float fine_speck_visibility =
      fade_px(fine_feature_size, fine_pixel_footprint);
  float2 speck_coordinates = (warped_coordinates * 48.0) + float2(2.0);
  float coarse_speckle =
      smoothstep(0.71 - transition_width, 0.71 + transition_width,
                 vnoise(speck_coordinates)) *
      speck_visibility;
  float2 fine_coordinates = (warped_coordinates * 110.0) + float2(9.0);
  float fine_speckle =
      smoothstep(0.78 - transition_width, 0.78 + transition_width,
                 vnoise(fine_coordinates)) *
      fine_speck_visibility;
  float paint_coverage = (((paint_islands * 0.3) + (paint_patches * 0.3)) +
                          (coarse_speckle * 0.25)) +
                         (fine_speckle * 0.15);
  paint_coverage = floor((paint_coverage * 4.0) + 0.5) / 4.0;
  float2 roughness_x_coordinates = warped_coordinates * float2(150.0, 38.0);
  float2 roughness_y_coordinates =
      (warped_coordinates * float2(38.0, 150.0)) + float2(4.0);
  float surface_roughness =
      ((vnoise(roughness_x_coordinates) + vnoise(roughness_y_coordinates)) -
       1.0) *
      fine_speck_visibility;
  float spray_tone =
      (0.95 - (paint_coverage * 0.17)) + (surface_roughness * 0.025);
  return float3(spray_tone * 0.995, spray_tone, spray_tone * 1.005);
}

static inline __attribute__((always_inline)) float3
m_papier(thread const float2& texture_coordinates,
         thread const float& pixel_footprint) {
  float relief_frequency = 22.0;
  float relief_sample_step = 0.35 / relief_frequency;
  float2 center_height_coordinates = texture_coordinates * relief_frequency;
  float center_height = fbm2(center_height_coordinates);
  float2 offset_height_x_coordinates =
      (texture_coordinates + float2(relief_sample_step, 0.0)) *
      relief_frequency;
  float offset_height_x = fbm2(offset_height_x_coordinates);
  float2 offset_height_y_coordinates =
      (texture_coordinates + float2(0.0, relief_sample_step)) *
      relief_frequency;
  float offset_height_y = fbm2(offset_height_y_coordinates);
  float relief_feature_size = 1.0 / relief_frequency;
  float relief_pixel_footprint = pixel_footprint;
  float paper_relief =
      (((center_height - offset_height_x) + (center_height - offset_height_y)) *
       2.2) *
      fade_px(relief_feature_size, relief_pixel_footprint);
  float2 tooth_coordinates = texture_coordinates * 140.0;
  float tooth_feature_size = 0.007142857;
  float tooth_pixel_footprint = pixel_footprint;
  float paper_tooth = (vnoise(tooth_coordinates) - 0.5) *
                      fade_px(tooth_feature_size, tooth_pixel_footprint);
  float2 tone_noise_coordinates = texture_coordinates * 1.1;
  float paper_tone =
      (((0.968 + (paper_relief * 0.05)) + (paper_tooth * 0.015)) +
       ((vnoise(tone_noise_coordinates) - 0.5) * 0.01)) *
      0.91;
  return float3(paper_tone, paper_tone * 0.997, paper_tone * 0.99);
}

static inline __attribute__((always_inline)) float3
material2(thread const int& material_index, thread float2& texture_coordinates,
          thread float& pixel_footprint) {
  int clamped_material_index = clamp(material_index, 0, 4);
  float texture_scale = material_texture_scales[clamped_material_index];
  texture_coordinates *= texture_scale;
  pixel_footprint *= texture_scale;
  float3 material_color;

  if (material_index == 1) {
    float2 cloth_coordinates = texture_coordinates;
    float cloth_pixel_footprint = pixel_footprint;
    material_color = m_stoff(cloth_coordinates, cloth_pixel_footprint);
  } else {
    if (material_index == 2) {
      float2 cardboard_coordinates = texture_coordinates;
      float cardboard_pixel_footprint = pixel_footprint;
      material_color =
          m_pappe(cardboard_coordinates, cardboard_pixel_footprint);
    } else {
      if (material_index == 3) {
        float2 wood_coordinates = texture_coordinates;
        float wood_pixel_footprint = pixel_footprint;
        material_color = m_holz(wood_coordinates, wood_pixel_footprint);
      } else {
        if (material_index == 4) {
          float2 spray_coordinates = texture_coordinates;
          float spray_pixel_footprint = pixel_footprint;
          material_color =
              m_spruehfarbe(spray_coordinates, spray_pixel_footprint);
        } else {
          float2 paper_coordinates = texture_coordinates;
          float paper_pixel_footprint = pixel_footprint;
          material_color = m_papier(paper_coordinates, paper_pixel_footprint);
        }
      }
    }
  }

  return material_color * material_brightness_gains[clamped_material_index];
}

static inline __attribute__((always_inline)) float3 material_triplanar(
    thread const int& material_index, thread const float3& texture_coordinates,
    thread const float3& surface_normal, thread const float& pixel_footprint) {
  float3 projection_weights = powr(abs(surface_normal), float3(6.0));
  projection_weights /= float3(fast::max(
      (projection_weights.x + projection_weights.y) + projection_weights.z,
      1e-6));
  float3 blended_color = float3(0.0);

  if (projection_weights.x > 0.01) {
    int projection_x_material = material_index;
    float2 projection_x_coordinates = texture_coordinates.zy;
    float projection_x_footprint = pixel_footprint;
    float3 projection_x_color =
        material2(projection_x_material, projection_x_coordinates,
                  projection_x_footprint);
    blended_color += (projection_x_color * projection_weights.x);
  }

  if (projection_weights.y > 0.01) {
    int projection_y_material = material_index;
    float2 projection_y_coordinates = texture_coordinates.xz + float2(13.7);
    float projection_y_footprint = pixel_footprint;
    float3 projection_y_color =
        material2(projection_y_material, projection_y_coordinates,
                  projection_y_footprint);
    blended_color += (projection_y_color * projection_weights.y);
  }

  if (projection_weights.z > 0.01) {
    int projection_z_material = material_index;
    float2 projection_z_coordinates = texture_coordinates.xy + float2(29.3);
    float projection_z_footprint = pixel_footprint;
    float3 projection_z_color =
        material2(projection_z_material, projection_z_coordinates,
                  projection_z_footprint);
    blended_color += (projection_z_color * projection_weights.z);
  }

  return blended_color /
         float3(fast::max(
             ((projection_weights.x * float(projection_weights.x > 0.01)) +
              (projection_weights.y * float(projection_weights.y > 0.01))) +
                 (projection_weights.z * float(projection_weights.z > 0.01)),
             1e-6));
}

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

static inline __attribute__((always_inline)) float fbm3(
    thread const float3& noise_coordinates) {
  float3 octave0_coordinates = noise_coordinates;
  float3 octave1_coordinates = (noise_coordinates * 2.03) + float3(7.1);
  float3 octave2_coordinates = (noise_coordinates * 4.01) + float3(3.7);
  float3 octave3_coordinates = (noise_coordinates * 8.07) + float3(1.3);
  return (((vnoise3(octave0_coordinates) * 0.5) +
           (vnoise3(octave1_coordinates) * 0.25)) +
          (vnoise3(octave2_coordinates) * 0.125)) +
         (vnoise3(octave3_coordinates) * 0.0625);
}

static inline __attribute__((always_inline)) float3
stains(thread const float3& texture_coordinates,
       thread const float& stain_strength) {
  if (stain_strength <= 0.0) {
    return float3(1.0);
  }

  float3 stain_coordinates = texture_coordinates * 0.3;
  float stain_noise = fbm3(stain_coordinates);
  float stain_body = smoothstep(0.5225, 0.6075, stain_noise);
  float stain_rim = smoothstep(0.5025, 0.5275, stain_noise) *
                    (1.0 - smoothstep(0.5325, 0.5825, stain_noise));
  float3 mottled_coordinates = (texture_coordinates * 1.3) + float3(31.0);
  float mottled_noise = fbm3(mottled_coordinates);
  float stain_amount = (((stain_body * (0.5 + (0.5 * mottled_noise))) * 0.12) +
                        (stain_rim * 0.09)) *
                       stain_strength;
  return mix(float3(1.0), float3(0.8, 0.7, 0.55), float3(stain_amount));
}

static inline __attribute__((always_inline)) float3
tone_steps(thread const float3& material_color) {
  float luminance = dot(material_color, float3(0.2126, 0.7152, 0.0722));
  float quantized_luminance = (floor(luminance / 0.022) + 0.5) * 0.022;
  return material_color * (quantized_luminance / fast::max(luminance, 0.0001));
}

static inline __attribute__((always_inline)) float band(
    thread const float& light_value, constant MaterialUniforms& uniforms) {
  if (uniforms.bands < 1.5) {
    return light_value;
  }

  return floor((fast::clamp(light_value, 0.0, 1.0) * (uniforms.bands - 1.0)) +
               0.5) /
         (uniforms.bands - 1.0);
}

static inline __attribute__((always_inline)) float in_shadow(
    thread const float3& world_position, constant MaterialUniforms& uniforms,
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
    thread const float3& sky_color, constant MaterialUniforms& uniforms) {
  float hemisphere_factor = 0.5 + (0.5 * surface_normal.y);
  float hemisphere_light = band(hemisphere_factor, uniforms);
  return mix(float3(direct_light),
             mix(float3(1.0), sky_color, float3(0.5)) * hemisphere_light,
             float3(uniforms.dome));
}

fragment material_frag_out
material_frag(material_frag_in stage_input [[stage_in]],
              constant MaterialUniforms& uniforms [[buffer(0)]],
              texture2d<float> surface_table [[texture(0)]],
              texture2d<float> shadow_map [[texture(1)]],
              texture2d<float> occlusion_accumulation [[texture(2)]],
              texture2d<float> occlusion_map [[texture(3)]],
              sampler surface_table_sampler [[sampler(0)]],
              sampler shadow_sampler [[sampler(1)]],
              sampler occlusion_accumulation_sampler [[sampler(2)]],
              sampler occlusion_sampler [[sampler(3)]],
              float4 fragment_position [[position]]) {
  material_frag_out stage_output = {};
  float3 surface_tint;
  int material_index;
  float3 texture_anchor;
  float texture_scale;
  float3 texture_offset;
  float grid_spacing;
  float4 selection_state;
  float3 grid_color;
  float grid_opacity;
  float shadow_lift;
  float surface_opacity;
  float4 owner_ids;
  load_surface(surface_table, surface_table_sampler, uniforms,
               stage_input.v_slot, surface_tint, material_index, texture_anchor,
               texture_scale, texture_offset, grid_spacing, selection_state,
               grid_color, grid_opacity, shadow_lift, surface_opacity,
               owner_ids);
  float3 normalized_vertex_normal;

  if (length(stage_input.v_normal) > 1e-9) {
    normalized_vertex_normal = fast::normalize(stage_input.v_normal);
  } else {
    normalized_vertex_normal = float3(0.0, 1.0, 0.0);
  }

  float3 vertex_normal_direction = normalized_vertex_normal;
  float3 normal_derivative_cross =
      cross(dfdx(stage_input.v_world), dfdy(stage_input.v_world));
  float3 normalized_geometric_normal;

  if (length(normal_derivative_cross) > 1e-20) {
    normalized_geometric_normal = fast::normalize(normal_derivative_cross);
  } else {
    normalized_geometric_normal = vertex_normal_direction;
  }

  float3 geometric_normal = normalized_geometric_normal;
  float3 texture_coordinates =
      ((stage_input.v_world - texture_anchor) / float3(texture_scale)) +
      texture_offset;
  float pixel_footprint =
      fast::min(length(fwidth(texture_coordinates)), 1000.0);
  float3 grid_coordinates =
      stage_input.v_world / float3(fast::max(grid_spacing, 1e-9));
  float3 grid_pixel_width = fast::max(fwidth(grid_coordinates), float3(1e-6));
  float3 coverage_position = stage_input.v_world;
  float4 coverage_owner_ids = owner_ids;
  float visibility =
      cover_edge(coverage_position, coverage_owner_ids, uniforms,
                 occlusion_accumulation, occlusion_accumulation_sampler,
                 occlusion_map, occlusion_sampler, fragment_position);
  bool is_front_layer = visibility >= 1.0;

  if ((is_front_layer != (uniforms.layer > 0.5)) || (visibility <= 0.0)) {
    discard_fragment();
  }

  float3 selected_view_direction;

  if (uniforms.ortho > 0.5) {
    selected_view_direction = float3(uniforms.view_back);
  } else {
    selected_view_direction =
        fast::normalize(uniforms.eye - stage_input.v_world);
  }

  float3 view_direction = selected_view_direction;
  int texture_material = material_index;
  float3 texture_sample_position = texture_coordinates;
  float3 texture_sample_normal = geometric_normal;
  float texture_pixel_footprint = pixel_footprint;
  float3 stain_sample_position = texture_coordinates;
  float stain_strength =
      uniforms.stain * material_stain_weights[clamp(material_index, 0, 4)];
  float3 stained_material_color =
      material_triplanar(texture_material, texture_sample_position,
                         texture_sample_normal, texture_pixel_footprint) *
      stains(stain_sample_position, stain_strength);
  float3 surface_color = tone_steps(stained_material_color) * surface_tint;
  float3 facing_normal = vertex_normal_direction;

  if (dot(facing_normal, view_direction) < 0.0) {
    facing_normal = -facing_normal;
  }

  float view_facing = abs(dot(facing_normal, view_direction));

  if (uniforms.light_on > 0.5) {
    float3 normalized_smooth_normal;

    if (length(stage_input.v_calm) > 1e-9) {
      normalized_smooth_normal = fast::normalize(stage_input.v_calm);
    } else {
      normalized_smooth_normal = facing_normal;
    }

    float3 smooth_normal = normalized_smooth_normal;

    if (dot(smooth_normal, facing_normal) < 0.0) {
      smooth_normal = -smooth_normal;
    }

    float3 lighting_normal = fast::normalize(
        mix(facing_normal, smooth_normal, float3(uniforms.calm)));
    float sun_facing = dot(lighting_normal, uniforms.sun_dir);
    float sun_illumination_input = smoothstep(-0.25, 0.9, sun_facing);
    float sun_illumination = band(sun_illumination_input, uniforms);
    float3 shadow_offset_normal;

    if (sun_facing >= 0.0) {
      shadow_offset_normal = lighting_normal;
    } else {
      shadow_offset_normal = -lighting_normal;
    }

    float3 shadow_query_position =
        (stage_input.v_world + (shadow_offset_normal * shadow_lift)) +
        (uniforms.sun_dir * shadow_lift);
    float shadow_sample =
        in_shadow(shadow_query_position, uniforms, shadow_map, shadow_sampler);
    float shadow_coverage = band(shadow_sample, uniforms);
    float light_amount =
        fast::min(mix(1.0 - uniforms.away_dark, 1.0, sun_illumination),
                  1.0 - (uniforms.shadow_dark * shadow_coverage));
    float3 lighting_sample_normal = lighting_normal;
    float lighting_sample_amount = light_amount;
    float3 sky_sample_color = float3(uniforms.sky_col);
    surface_color *=
        (sun_and_sky(lighting_sample_normal, lighting_sample_amount,
                     sky_sample_color, uniforms) *
         mix(float3(uniforms.sky_col), uniforms.sun_col,
             float3(sun_illumination * (1.0 - shadow_coverage))));
  } else {
    surface_color *= (0.9 + (0.1 * view_facing));
  }

  if ((grid_spacing > 0.0) && (grid_opacity > 0.0)) {
    float3 grid_edge_distance =
        abs(fract(grid_coordinates - float3(0.5)) - float3(0.5)) /
        grid_pixel_width;
    float3 grid_projection_weights = powr(abs(geometric_normal), float3(4.0));
    grid_projection_weights /= float3(
        fast::max((grid_projection_weights.x + grid_projection_weights.y) +
                      grid_projection_weights.z,
                  1e-6));
    float grid_line_x =
        1.0 -
        fast::min(fast::min(grid_edge_distance.y, grid_edge_distance.z), 1.0);
    float grid_line_y =
        1.0 -
        fast::min(fast::min(grid_edge_distance.x, grid_edge_distance.z), 1.0);
    float grid_line_z =
        1.0 -
        fast::min(fast::min(grid_edge_distance.x, grid_edge_distance.y), 1.0);
    surface_color = mix(surface_color, grid_color,
                        float3((((grid_projection_weights.x * grid_line_x) +
                                 (grid_projection_weights.y * grid_line_y)) +
                                (grid_projection_weights.z * grid_line_z)) *
                               grid_opacity));
  }

  surface_color =
      mix(surface_color, selection_state.xyz, float3(selection_state.w));
  float3 luminance_weights = float3(0.2126, 0.7152, 0.0722);
  surface_color =
      mix(surface_color, float3(dot(surface_color, luminance_weights)),
          float3(uniforms.mono));
  float layer_opacity;

  if (is_front_layer) {
    layer_opacity = uniforms.front_alpha;
  } else {
    layer_opacity = visibility;
  }

  stage_output.f_color = float4(surface_color, layer_opacity * surface_opacity);
  return stage_output;
}

static_assert(sizeof(MaterialUniforms) == 352, "Uniform size");
static_assert(__builtin_offsetof(MaterialUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(MaterialUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(MaterialUniforms, occluders_only) == 68,
              "occluders_only offset");
static_assert(__builtin_offsetof(MaterialUniforms, surf_rows) == 72,
              "surf_rows offset");
static_assert(__builtin_offsetof(MaterialUniforms, lift_min) == 76,
              "lift_min offset");
static_assert(__builtin_offsetof(MaterialUniforms, stain) == 80,
              "stain offset");
static_assert(__builtin_offsetof(MaterialUniforms, layer) == 84,
              "layer offset");
static_assert(__builtin_offsetof(MaterialUniforms, front_alpha) == 88,
              "front_alpha offset");
static_assert(__builtin_offsetof(MaterialUniforms, mono) == 92, "mono offset");
static_assert(__builtin_offsetof(MaterialUniforms, light_on) == 96,
              "light_on offset");
static_assert(__builtin_offsetof(MaterialUniforms, sun_dir) == 112,
              "sun_dir offset");
static_assert(__builtin_offsetof(MaterialUniforms, sun_col) == 128,
              "sun_col offset");
static_assert(__builtin_offsetof(MaterialUniforms, sky_col) == 144,
              "sky_col offset");
static_assert(__builtin_offsetof(MaterialUniforms, away_dark) == 156,
              "away_dark offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_dark) == 160,
              "shadow_dark offset");
static_assert(__builtin_offsetof(MaterialUniforms, calm) == 164, "calm offset");
static_assert(__builtin_offsetof(MaterialUniforms, bands) == 168,
              "bands offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_on) == 172,
              "shadow_on offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_vp) == 176,
              "shadow_vp offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_bias) == 240,
              "shadow_bias offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_lift) == 244,
              "shadow_lift offset");
static_assert(__builtin_offsetof(MaterialUniforms, dome) == 248, "dome offset");
static_assert(__builtin_offsetof(MaterialUniforms, eye) == 256, "eye offset");
static_assert(__builtin_offsetof(MaterialUniforms, view_back) == 272,
              "view_back offset");
static_assert(__builtin_offsetof(MaterialUniforms, ortho) == 284,
              "ortho offset");
static_assert(__builtin_offsetof(MaterialUniforms, occl_on) == 288,
              "occl_on offset");
static_assert(__builtin_offsetof(MaterialUniforms, xray_behind) == 292,
              "xray_behind offset");
static_assert(__builtin_offsetof(MaterialUniforms, occl_tol) == 304,
              "occl_tol offset");
static_assert(__builtin_offsetof(MaterialUniforms, occl_min) == 316,
              "occl_min offset");
static_assert(__builtin_offsetof(MaterialUniforms, aid_pass) == 320,
              "aid_pass offset");
static_assert(__builtin_offsetof(MaterialUniforms, line_info) == 336,
              "line_info offset");
