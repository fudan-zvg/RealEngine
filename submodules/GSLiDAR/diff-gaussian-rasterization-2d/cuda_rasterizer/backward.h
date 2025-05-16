/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2 *ranges,
		const uint32_t *point_list,
		const int S, int W, int H,
		const float *bg_color,
		const float2 *means2D,
		const float4 *normal_opacity,
		const float *colors,
		const float *transMats,
		const float *depths,
		const float *flows_2d,
		const float *final_Ts,
		const int32_t *n_contrib,
		const float *dL_dpixels,
		const float *dL_depths,
		const float *dL_masks,
		const float *dL_dpix_flow,
		float *dL_dtransMat,
		float4 *dL_dmean2D,
		float *dL_dopacity,
		float *dL_dcolors,
		float *dL_dflows,
		float *dL_dnormals,
		const float vfov_min,
		const float vfov_max,
		const float hfov_min,
		const float hfov_max,
		const float scale_factor);

	void preprocess(
		int P, int D, int M,
		const float3 *means,
		const int *radii,
		const float *shs,
		const bool *clamped,
		const glm::vec3 *scales,
		const glm::vec4 *rotations,
		const float scale_modifier,
		const float *transMats,
		const float *view,
		const float *proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3 *campos,
		float4 *dL_dmean2D,
		float* dL_dnormals,
		float *dL_dtransMats,
		glm::vec3 *dL_dmeans,
		float *dL_dcolor,
		float *dL_dsh,
		glm::vec3 *dL_dscale,
		glm::vec4 *dL_drot,
		const float vfov_min,
		const float vfov_max,
		const float hfov_min,
		const float hfov_max,
		const int width, int height);
}

#endif