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

struct resolve8_frag_out {
  float4 f_color [[color(0)]];
};

fragment resolve8_frag_out resolve8_frag(texture2d_ms<float> source_texture
                                         [[texture(0)]],
                                         sampler source_sampler [[sampler(0)]],
                                         float4 fragment_position
                                         [[position]]) {
  resolve8_frag_out stage_output = {};
  int2 sample_pixel = int2(fragment_position.xy);
  float4 sample_sum = float4(0.0);
  for (int sample_index = 0; sample_index < 8; sample_index++) {
    sample_sum += source_texture.read(uint2(sample_pixel), sample_index);
  }

  stage_output.f_color = sample_sum / float4(8.0);
  return stage_output;
}
