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

struct PictureUniforms {
  float4x4 vp;
  float bias0;
  char padding0[12];
  float4x4 picture_m;
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float4 owner;
  float under;
  float occl_on;
  float xray_behind;
  char padding1[4];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding2[12];
  float4 line_info;
};

struct picture_frag_out {
  float4 f_color [[color(0)]];
};

struct picture_frag_in {
  float3 v_world [[user(locn0)]];
};

static inline __attribute__((always_inline)) bool occl_dense_at(
    thread const int2& pixel_coordinates, constant PictureUniforms& uniforms,
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
    thread int2& nearest_pixel, constant PictureUniforms& uniforms,
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
    thread const float3& world_position, constant PictureUniforms& uniforms) {
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
    constant PictureUniforms& uniforms, texture2d<float> occlusion_accumulation,
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
    constant PictureUniforms& uniforms, texture2d<float> occlusion_accumulation,
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

fragment picture_frag_out
picture_frag(picture_frag_in stage_input [[stage_in]],
             constant PictureUniforms& uniforms [[buffer(0)]],
             texture2d<float> occlusion_accumulation [[texture(0)]],
             texture2d<float> occlusion_map [[texture(1)]],
             texture2d<float> picture_texture [[texture(2)]],
             sampler occlusion_accumulation_sampler [[sampler(0)]],
             sampler occlusion_sampler [[sampler(1)]],
             sampler picture_sampler [[sampler(2)]],
             float4 fragment_position [[position]]) {
  picture_frag_out stage_output = {};
  float4 picture_coordinates =
      uniforms.picture_m * float4(stage_input.v_world, 1.0);

  if (picture_coordinates.w <= 0.0) {
    discard_fragment();
  }

  float2 picture_uv =
      ((picture_coordinates.xy / float2(picture_coordinates.w)) * 0.5) +
      float2(0.5);
  float4 picture_color = picture_texture.sample(picture_sampler, picture_uv);
  bool outside_lower_uv = any(picture_uv < float2(0.0));
  bool outside_picture;

  if (!outside_lower_uv) {
    outside_picture = any(picture_uv > float2(1.0));
  } else {
    outside_picture = outside_lower_uv;
  }

  if (outside_picture) {
    discard_fragment();
  }

  float3 query_position = stage_input.v_world;
  float4 query_owner_ids = uniforms.owner;
  float visibility =
      cover(query_position, query_owner_ids, uniforms, occlusion_accumulation,
            occlusion_accumulation_sampler, fragment_position, occlusion_map,
            occlusion_sampler);
  if (visibility <= 0.0) {
    discard_fragment();
  }

  stage_output.f_color =
      float4(picture_color.xyz, picture_color.w * visibility);
  return stage_output;
}

static_assert(sizeof(PictureUniforms) == 256, "Uniform size");
static_assert(__builtin_offsetof(PictureUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(PictureUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(PictureUniforms, picture_m) == 80,
              "picture_m offset");
static_assert(__builtin_offsetof(PictureUniforms, eye) == 144, "eye offset");
static_assert(__builtin_offsetof(PictureUniforms, view_back) == 160,
              "view_back offset");
static_assert(__builtin_offsetof(PictureUniforms, ortho) == 172,
              "ortho offset");
static_assert(__builtin_offsetof(PictureUniforms, owner) == 176,
              "owner offset");
static_assert(__builtin_offsetof(PictureUniforms, under) == 192,
              "under offset");
static_assert(__builtin_offsetof(PictureUniforms, occl_on) == 196,
              "occl_on offset");
static_assert(__builtin_offsetof(PictureUniforms, xray_behind) == 200,
              "xray_behind offset");
static_assert(__builtin_offsetof(PictureUniforms, occl_tol) == 208,
              "occl_tol offset");
static_assert(__builtin_offsetof(PictureUniforms, occl_min) == 220,
              "occl_min offset");
static_assert(__builtin_offsetof(PictureUniforms, aid_pass) == 224,
              "aid_pass offset");
static_assert(__builtin_offsetof(PictureUniforms, line_info) == 240,
              "line_info offset");
