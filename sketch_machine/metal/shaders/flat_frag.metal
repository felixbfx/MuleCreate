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

struct FlatUniforms {
  float4x4 vp;
  float bias0;
  char padding0[12];
  float4 color;
  float grain;
  char padding1[12];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float milk;
  char padding2[12];
  float4 owner;
  float under;
  float soft;
  float occl_on;
  float xray_behind;
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding3[12];
  float4 line_info;
};

struct flat_frag_out {
  float4 f_color [[color(0)]];
};

struct flat_frag_in {
  float3 v_world [[user(locn0)]];
};

static inline __attribute__((always_inline)) bool occl_dense_at(
    thread const int2& pixel_coordinates, constant FlatUniforms& uniforms,
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

static inline __attribute__((always_inline)) float4 occl_front(
    thread int2& nearest_pixel, constant FlatUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, thread float4& fragment_position,
    texture2d<float> occlusion_map, sampler occlusion_sampler) {
  int2 fragment_pixel = int2(fragment_position.xy);
  int2 occlusion_size =
      int2(occlusion_map.get_width(), occlusion_map.get_height());
  float4 nearest_occluder = occlusion_map.read(uint2(fragment_pixel), 0);
  nearest_pixel = fragment_pixel;
  for (int sample_offset_x = -1; sample_offset_x <= 1; sample_offset_x++) {
    for (int sample_offset_y = -1; sample_offset_y <= 1; sample_offset_y++) {
      int2 sample_pixel =
          clamp(fragment_pixel + int2(sample_offset_x, sample_offset_y),
                int2(0), occlusion_size - int2(1));
      float4 sample_occluder = occlusion_map.read(uint2(sample_pixel), 0);
      bool has_occluder = sample_occluder.z > 0.0;
      bool is_nearer_candidate;

      if (has_occluder) {
        bool previous_sample_empty = nearest_occluder.z <= 0.0;
        bool candidate_is_nearer;

        if (!previous_sample_empty) {
          candidate_is_nearer = sample_occluder.x < nearest_occluder.x;
        } else {
          candidate_is_nearer = previous_sample_empty;
        }

        is_nearer_candidate = candidate_is_nearer;
      } else {
        is_nearer_candidate = has_occluder;
      }

      bool candidate_is_dense;

      if (is_nearer_candidate) {
        int2 density_pixel = sample_pixel;
        candidate_is_dense =
            occl_dense_at(density_pixel, uniforms, occlusion_accumulation,
                          occlusion_accumulation_sampler);
      } else {
        candidate_is_dense = is_nearer_candidate;
      }

      if (candidate_is_dense) {
        nearest_occluder = sample_occluder;
        nearest_pixel = sample_pixel;
      }
    }
  }

  return nearest_occluder;
}

static inline __attribute__((always_inline)) float view_dist(
    thread const float3& world_position, constant FlatUniforms& uniforms) {
  float distance_to_eye;

  if (uniforms.ortho > 0.5) {
    distance_to_eye =
        dot(uniforms.eye - world_position, float3(uniforms.view_back));
  } else {
    distance_to_eye = length(world_position - uniforms.eye);
  }

  return distance_to_eye;
}

static inline __attribute__((always_inline)) float cover_t(
    thread const float3& world_position, thread const float4& owner_ids,
    thread const float& foreign_owner_tolerance,
    constant FlatUniforms& uniforms, texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, thread float4& fragment_position,
    texture2d<float> occlusion_map, sampler occlusion_sampler) {
  if (uniforms.occl_on < 0.5) {
    return 1.0;
  }

  int2 nearest_pixel;
  float4 nearest_front_occluder =
      occl_front(nearest_pixel, uniforms, occlusion_accumulation,
                 occlusion_accumulation_sampler, fragment_position,
                 occlusion_map, occlusion_sampler);
  int2 front_pixel = nearest_pixel;
  float4 front_occluder = nearest_front_occluder;
  bool front_sample_empty = front_occluder.z <= 0.0;
  bool front_sample_transparent;

  if (!front_sample_empty) {
    int2 density_pixel = front_pixel;
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
      foreign_tolerance = foreign_owner_tolerance;
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
                   exp(-occlusion_accumulation.read(uint2(front_pixel), 0).x));
}

static inline __attribute__((always_inline)) float cover(
    thread const float3& world_position, thread const float4& owner_ids,
    constant FlatUniforms& uniforms, texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, thread float4& fragment_position,
    texture2d<float> occlusion_map, sampler occlusion_sampler) {
  float3 query_position = world_position;
  float4 query_owner_ids = owner_ids;
  float foreign_owner_tolerance = uniforms.occl_tol[2u];
  return cover_t(query_position, query_owner_ids, foreign_owner_tolerance,
                 uniforms, occlusion_accumulation,
                 occlusion_accumulation_sampler, fragment_position,
                 occlusion_map, occlusion_sampler);
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

static inline __attribute__((always_inline)) float paper2(
    thread const float2& texture_coordinates,
    thread const float& pixel_footprint) {
  float2 blotch_coordinates = texture_coordinates * 0.7;
  float blotch_noise = vnoise(blotch_coordinates) - 0.5;
  float2 horizontal_fiber_coordinates = texture_coordinates * float2(26.0, 5.0);
  float2 vertical_fiber_coordinates =
      (texture_coordinates * float2(5.0, 26.0)) + float2(41.0);
  float fiber_noise = ((vnoise(horizontal_fiber_coordinates) +
                        vnoise(vertical_fiber_coordinates)) *
                       0.5) -
                      0.5;
  float2 tooth_coordinates = texture_coordinates * 70.0;
  float tooth_noise = vnoise(tooth_coordinates) - 0.5;
  fiber_noise *= (1.0 - smoothstep(0.02, 0.08, pixel_footprint));
  tooth_noise *= (1.0 - smoothstep(0.008, 0.03, pixel_footprint));
  return ((blotch_noise * 0.04) + (fiber_noise * 0.6)) + (tooth_noise * 0.7);
}

static inline __attribute__((always_inline)) float paper_triplanar(
    thread const float3& world_position, thread const float3& surface_normal,
    thread const float& pixel_footprint) {
  float3 projection_weights = powr(abs(surface_normal), float3(4.0));
  projection_weights /= float3(fast::max(
      (projection_weights.x + projection_weights.y) + projection_weights.z,
      1e-6));
  float2 projection_x_coordinates = world_position.yz;
  float projection_x_footprint = pixel_footprint;
  float2 projection_y_coordinates = world_position.zx + float2(13.0);
  float projection_y_footprint = pixel_footprint;
  float2 projection_z_coordinates = world_position.xy + float2(29.0);
  float projection_z_footprint = pixel_footprint;
  return ((paper2(projection_x_coordinates, projection_x_footprint) *
           projection_weights.x) +
          (paper2(projection_y_coordinates, projection_y_footprint) *
           projection_weights.y)) +
         (paper2(projection_z_coordinates, projection_z_footprint) *
          projection_weights.z);
}

static inline __attribute__((always_inline)) bool on_layer(
    thread const float3& world_position, constant FlatUniforms& uniforms,
    thread float4& fragment_position, texture2d<float> occlusion_map,
    sampler occlusion_sampler) {
  if (uniforms.occl_on < 0.5) {
    return false;
  }

  float4 front_occluder =
      occlusion_map.read(uint2(int2(fragment_position.xy)), 0);
  bool has_front_layer = front_occluder.z > 0.0;
  bool on_or_behind_front_layer;

  if (has_front_layer) {
    float3 query_position = world_position;
    on_or_behind_front_layer =
        view_dist(query_position, uniforms) >
        (front_occluder.x - (uniforms.occl_tol[0u] * front_occluder.z));
  } else {
    on_or_behind_front_layer = has_front_layer;
  }

  return on_or_behind_front_layer;
}

static inline __attribute__((always_inline)) float front_opacity(
    constant FlatUniforms& uniforms, texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, thread float4& fragment_position,
    texture2d<float> occlusion_map, sampler occlusion_sampler) {
  if (uniforms.occl_on < 0.5) {
    return 0.0;
  }

  if (occlusion_map.read(uint2(int2(fragment_position.xy)), 0).z <= 0.0) {
    return 0.0;
  }

  return 1.0 -
         exp(-occlusion_accumulation.read(uint2(int2(fragment_position.xy)), 0)
                  .x);
}

fragment flat_frag_out flat_frag(flat_frag_in stage_input [[stage_in]],
                                 constant FlatUniforms& uniforms [[buffer(0)]],
                                 texture2d<float> occlusion_accumulation
                                 [[texture(0)]],
                                 texture2d<float> occlusion_map [[texture(1)]],
                                 sampler occlusion_accumulation_sampler
                                 [[sampler(0)]],
                                 sampler occlusion_sampler [[sampler(1)]],
                                 float4 fragment_position [[position]]) {
  flat_frag_out stage_output = {};
  float3 normal_derivative_cross =
      cross(dfdx(stage_input.v_world), dfdy(stage_input.v_world));
  float3 normalized_surface_normal;

  if (length(normal_derivative_cross) > 1e-20) {
    normalized_surface_normal = fast::normalize(normal_derivative_cross);
  } else {
    normalized_surface_normal = float3(0.0, 1.0, 0.0);
  }

  float3 surface_normal = normalized_surface_normal;
  float world_pixel_footprint = length(fwidth(stage_input.v_world));
  float3 query_position = stage_input.v_world;
  float4 query_owner_ids = uniforms.owner;
  float visibility =
      cover(query_position, query_owner_ids, uniforms, occlusion_accumulation,
            occlusion_accumulation_sampler, fragment_position, occlusion_map,
            occlusion_sampler);
  if (visibility <= 0.0) {
    discard_fragment();
  }

  float paper_texture = 0.0;
  float plane_opacity = uniforms.color.w;

  if (uniforms.grain > 0.0) {
    float3 texture_world_position = stage_input.v_world;
    float3 texture_normal = surface_normal;
    float texture_pixel_footprint = world_pixel_footprint;
    paper_texture = paper_triplanar(texture_world_position, texture_normal,
                                    texture_pixel_footprint);
    float3 selected_view_direction;

    if (uniforms.ortho > 0.5) {
      selected_view_direction = float3(uniforms.view_back);
    } else {
      selected_view_direction =
          fast::normalize(uniforms.eye - stage_input.v_world);
    }

    float3 view_direction = selected_view_direction;
    plane_opacity = fast::min(
        plane_opacity *
            (1.0 + (uniforms.milk *
                    (1.0 - abs(dot(surface_normal, view_direction))))),
        1.0);
  }

  float3 layer_query_position = stage_input.v_world;
  bool plane_behind_surface =
      on_layer(layer_query_position, uniforms, fragment_position, occlusion_map,
               occlusion_sampler);

  if (uniforms.aid_pass < 0.5) {
    if (!plane_behind_surface) {
      discard_fragment();
    }
  } else {
    if (plane_behind_surface) {
      if (uniforms.under <= 0.0) {
        discard_fragment();
      }

      plane_opacity = uniforms.under;
    } else {
      if (uniforms.under <= 0.0) {
        plane_opacity *= (1.0 - front_opacity(uniforms, occlusion_accumulation,
                                              occlusion_accumulation_sampler,
                                              fragment_position, occlusion_map,
                                              occlusion_sampler));
      } else {
        if (uniforms.soft > 0.5) {
          plane_opacity = mix(
              plane_opacity, uniforms.under,
              front_opacity(uniforms, occlusion_accumulation,
                            occlusion_accumulation_sampler, fragment_position,
                            occlusion_map, occlusion_sampler));
        }
      }
    }
  }

  if (plane_opacity <= 0.0) {
    discard_fragment();
  }

  stage_output.f_color = float4(
      uniforms.color.xyz * (1.0 + ((uniforms.grain * 0.06) * paper_texture)),
      plane_opacity * visibility);
  return stage_output;
}

static_assert(sizeof(FlatUniforms) == 240, "Uniform size");
static_assert(__builtin_offsetof(FlatUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(FlatUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(FlatUniforms, color) == 80, "color offset");
static_assert(__builtin_offsetof(FlatUniforms, grain) == 96, "grain offset");
static_assert(__builtin_offsetof(FlatUniforms, eye) == 112, "eye offset");
static_assert(__builtin_offsetof(FlatUniforms, view_back) == 128,
              "view_back offset");
static_assert(__builtin_offsetof(FlatUniforms, ortho) == 140, "ortho offset");
static_assert(__builtin_offsetof(FlatUniforms, milk) == 144, "milk offset");
static_assert(__builtin_offsetof(FlatUniforms, owner) == 160, "owner offset");
static_assert(__builtin_offsetof(FlatUniforms, under) == 176, "under offset");
static_assert(__builtin_offsetof(FlatUniforms, soft) == 180, "soft offset");
static_assert(__builtin_offsetof(FlatUniforms, occl_on) == 184,
              "occl_on offset");
static_assert(__builtin_offsetof(FlatUniforms, xray_behind) == 188,
              "xray_behind offset");
static_assert(__builtin_offsetof(FlatUniforms, occl_tol) == 192,
              "occl_tol offset");
static_assert(__builtin_offsetof(FlatUniforms, occl_min) == 204,
              "occl_min offset");
static_assert(__builtin_offsetof(FlatUniforms, aid_pass) == 208,
              "aid_pass offset");
static_assert(__builtin_offsetof(FlatUniforms, line_info) == 224,
              "line_info offset");
