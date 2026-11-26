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

struct ShadowSurfaceUniforms {
  float4x4 vp;
  float bias0;
  int occluders_only;
  int surf_rows;
  char padding0[4];
  float4 line_info;
};

fragment void shadow_surface_frag() {}

static_assert(sizeof(ShadowSurfaceUniforms) == 96, "Uniform size");
static_assert(__builtin_offsetof(ShadowSurfaceUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(ShadowSurfaceUniforms, bias0) == 64,
              "bias offset");
static_assert(__builtin_offsetof(ShadowSurfaceUniforms, occluders_only) == 68,
              "occluders_only offset");
static_assert(__builtin_offsetof(ShadowSurfaceUniforms, surf_rows) == 72,
              "surf_rows offset");
static_assert(__builtin_offsetof(ShadowSurfaceUniforms, line_info) == 80,
              "line_info offset");
