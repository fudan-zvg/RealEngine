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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, const bool *clamped, const glm::vec4 *dL_dcolor, glm::vec3 *dL_dmeans, glm::vec4 *dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec4 *sh = ((glm::vec4 *)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec4 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[4 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[4 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[4 * idx + 2] ? 0 : 1;
	dL_dRGB.w *= clamped[4 * idx + 3] ? 0 : 1;

	glm::vec4 dRGBdx(0, 0, 0, 0);
	glm::vec4 dRGBdy(0, 0, 0, 0);
	glm::vec4 dRGBdz(0, 0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec4 *dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
						   SH_C3[1] * sh[10] * yz +
						   SH_C3[2] * sh[11] * -2.f * xy +
						   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
						   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
						   SH_C3[5] * sh[14] * 2.f * xz +
						   SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
						   SH_C3[1] * sh[10] * xz +
						   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
						   SH_C3[3] * sh[12] * -3.f * 2.f * yz +
						   SH_C3[4] * sh[13] * -2.f * xy +
						   SH_C3[5] * sh[14] * -2.f * yz +
						   SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (SH_C3[1] * sh[10] * xy +
						   SH_C3[2] * sh[11] * 4.f * 2.f * yz +
						   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
						   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
						   SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		const int S, int W, int H,
		const float *__restrict__ bg_color,
		const float2 *__restrict__ points_xy_image,
		const float4 *__restrict__ normal_opacity,
		const float *__restrict__ transMats,
		const float *__restrict__ colors,
		const float *__restrict__ depths,
		const float *__restrict__ features,
		const float *__restrict__ final_Ts,
		const int32_t *__restrict__ n_contrib,
		const float *__restrict__ dL_dpixels,
		const float *__restrict__ dL_depths,
		const float *__restrict__ dL_masks,
		const float *__restrict__ dL_dpix_feature,
		float *__restrict__ dL_dtransMat,
		float4 *__restrict__ dL_dmean2D,
		// float4 *__restrict__ dL_dconic2D,
		float *__restrict__ dL_dopacity,
		float *__restrict__ dL_dcolors,
		float *__restrict__ dL_dfeatures,
		float *__restrict__ dL_dnormals,
		const float vfov_min,
		const float vfov_max,
		const float hfov_min,
		const float hfov_max,
		const float scale_factor)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float collected_features[13 * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;

	float accum_rec[C] = {0};
	float accum_feature_rec[13 + 3] = {0};
	float dL_dpixel[C] = {0};
	float dL_dfeature[13] = {0};
	float dL_depth = 0;
	float dL_dmedian_depth = 0;
	float dL_ddistortion = 0;
	float dL_depth_square = 0;
	float dL_mask = 0;
	float accum_depth_rec = 0;
	float accum_mask_rec = 0;

	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;

	if (inside)
	{
		for (int i = 0; i < C; i++)
		{
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		}
		for (int i = 0; i < S + 3; i++)
		{
			dL_dfeature[i] = dL_dpix_feature[i * H * W + pix_id];
		}
		dL_depth = dL_depths[pix_id];
		dL_dmedian_depth = dL_depths[H * W + pix_id];
		dL_ddistortion = dL_depths[2 * H * W + pix_id];
		dL_depth_square = dL_depths[3 * H * W + pix_id];
		dL_mask = dL_masks[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = {0};
	float last_feature[13] = {0};
	float last_depth = 0;

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	// const float ddelx_dx = 0.5 * W;
	// const float ddely_dy = 0.5 * H;

	// vfov 角度制转弧度制
	const float VFOV_max = MY_PI / 2 - vfov_min * MY_PI / 180;
	const float VFOV_min = MY_PI / 2 - vfov_max * MY_PI / 180;

	// hfov 角度制转弧度制
	const float HFOV_max = hfov_max * MY_PI / 180;
	const float HFOV_min = hfov_min * MY_PI / 180;

	const float near = near_n * scale_factor;
	const float far = far_n * scale_factor;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_depths[block.thread_rank()] = depths[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			for (int i = 0; i < S; i++)
				collected_features[i * BLOCK_SIZE + block.thread_rank()] = features[coll_id * S + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];

			const float phi = pixf.x * (HFOV_max - HFOV_min) / W + HFOV_min;
			const float theta = pixf.y * (VFOV_max - VFOV_min) / H + VFOV_min;

			float3 k = cos(phi) * Tu - sin(phi) * Tw;
			float3 l = sin(phi) * cos(theta) * Tu + sin(theta) * Tv + cos(phi) * cos(theta) * Tw;
			float3 p = cross(k, l);
			if (p.z == 0.0)
				continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y);
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

			// if (rho3d > rho2d)
			// 	continue;

			float rho = min(rho3d, rho2d);
			// cos(theta) 会趋于0，导致不稳定
			// const float depth_3d_0 = -(s.x * Tv.x + s.y * Tv.y + Tv.z) / cos(theta);
			const float s_Tu = s.x * Tu.x + s.y * Tu.y + Tu.z;
			const float s_Tv = s.x * Tv.x + s.y * Tv.y + Tv.z;
			const float s_Tw = s.x * Tw.x + s.y * Tw.y + Tw.z;
			const float depth_3d = s_Tu * sin(theta) * sin(phi) - s_Tv * cos(theta) + s_Tw * sin(theta) * cos(phi);
			const float depth = (rho3d <= rho2d) ? depth_3d : collected_depths[j];
			if (depth < near || depth > far)
				continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			const float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			float dL_dr = 0.0f;
			dL_dr += alpha * T * dL_depth;
			dL_dr += alpha * T * 2 * depth * dL_depth_square;
			if (contributor == median_contributor - 1)
			{
				dL_dr += dL_dmedian_depth;
			}

			float dL_dweight = 0;
			const float m_d = far / (far - near) * (1 - near / depth);
			const float dmd_dd = (far * near) / ((far - near) * depth * depth);
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_ddistortion;
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_ddistortion;
			dL_dr += dL_dmd * dmd_dd;

			for (int ch = 0; ch < S + 3; ch++)
			{
				float feature;
				if (ch < S)
					feature = collected_features[ch * BLOCK_SIZE + j];
				else
					feature = normal[ch - S];

				// Update last color (to be used in the next iteration)
				accum_feature_rec[ch] = last_alpha * last_feature[ch] + (1.f - last_alpha) * accum_feature_rec[ch];
				last_feature[ch] = feature;

				const float dL_dchannelfeature = dL_dfeature[ch];
				// dL_dalpha += (feature - accum_feature_rec[ch]) * dL_dchannelfeature;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				if (ch < S)
					atomicAdd(&(dL_dfeatures[global_id * S + ch]), dchannel_dcolor * dL_dchannelfeature);
				else
				{
					dL_dalpha += (feature - accum_feature_rec[ch]) * dL_dchannelfeature;
					atomicAdd(&(dL_dnormals[global_id * 3 + ch - S]), dchannel_dcolor * dL_dchannelfeature); // normal loss
				}
			}

			// Propagate gradients to per-Gaussian depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = depth;
			dL_dalpha += ((depth - accum_depth_rec) * dL_depth);

			// Propagate gradients from masks
			accum_mask_rec = last_alpha + (1.f - last_alpha) * accum_mask_rec;
			dL_dalpha += ((1.0 - accum_mask_rec) * dL_mask);

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;

			if (rho3d <= rho2d)
			{
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dr * (Tu.x * sin(theta) * sin(phi) - Tv.x * cos(theta) + Tw.x * sin(theta) * cos(phi)),
					dL_dG * -G * s.y + dL_dr * (Tu.y * sin(theta) * sin(phi) - Tv.y * cos(theta) + Tw.y * sin(theta) * cos(phi))};
				const float3 dr_dTu = sin(theta) * sin(phi) * float3{s.x, s.y, 1.f};
				const float3 dr_dTv = -cos(theta) * float3{s.x, s.y, 1.0};
				const float3 dr_dTw = sin(theta) * cos(phi) * float3{s.x, s.y, 1.f};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = cos(phi) * dL_dk + sin(phi) * cos(theta) * dL_dl + dL_dr * dr_dTu;
				const float3 dL_dTv = sin(theta) * dL_dl + dL_dr * dr_dTv;
				const float3 dL_dTw = -sin(phi) * dL_dk + cos(phi) * cos(theta) * dL_dl + dL_dr * dr_dTw;

				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0], dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1], dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3], dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4], dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dTw.z);

				// const float u = transMats[global_id * 9 + 2];
				// const float v = transMats[global_id * 9 + 5];
				// const float w = transMats[global_id * 9 + 8];
				// const float dL_du = dL_dTu.z;
				// const float dL_dv = dL_dTv.z;
				// const float dL_dw = dL_dTw.z;

				// const float dx_du = w / (u * u + w * w);
				// const float dx_dw = -u / (u * u + w * w);
				// const float dL_dmean2D_x = (dL_du / dx_du + dL_dw / dx_dw) * 0.5 * (HFOV_max - HFOV_min);

				// const float dy_du = -u * v / (sqrt(u * u + w * w) * (u * u + v * v + w * w));
				// const float dy_dv = sqrt(u * u + w * w) / (u * u + v * v + w * w);
				// const float dy_dw = -w * v / (sqrt(u * u + w * w) * (u * u + v * v + w * w));
				// const float dL_dmean2D_y = (dL_du / dy_du + dL_dv / dy_dv + dL_dw / dy_dw) * 0.5 * (VFOV_max - VFOV_min);

				// const float phi = atan2f(u, w);
				// // const float theta = atan2f(sqrt(u * u + w * w), -v);
				// // const float r = sqrt(u * u + v * v + w * w);

				// const float du_dphi = w;  // r * sin(theta) * cos(phi)
				// const float dw_dphi = -u; // -r * sin(theta) * sin(phi)
				// const float dL_dmean2D_x = (dL_du * du_dphi + dL_dw * dw_dphi) * 0.5 * (HFOV_max - HFOV_min);

				// const float du_dtheta = -v * sin(phi);		 // r * cos(theta) * sin(phi)
				// const float dv_dtheta = sqrt(u * u + w * w); // r * sin(theta)
				// const float dw_dtheta = -v * cos(phi);		 // r * cos(theta) * cos(phi)
				// const float dL_dmean2D_y = (dL_du * du_dtheta + dL_dv * dv_dtheta + dL_dw * dw_dtheta) * 0.5 * (VFOV_max - VFOV_min);

				// // AbsGS densitify
				// atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dmean2D_x));
				// atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dmean2D_y));
			}
			else
			{
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;

				// Update gradients w.r.t. 2D mean position of the Gaussian
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dr * Tu.z / depth);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dr * Tv.z / depth);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dr * Tw.z / depth);

				// // AbsGS densitify
				// atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dG * dG_ddelx * 0.5 * W));
				// atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dG * dG_ddely * 0.5 * H));
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

__device__ void compute_transmat_aabb(
	int idx,
	const float3 *p_origs,
	const glm::vec3 *scales,
	const glm::vec4 *rots,
	const float *projmatrix,
	const float *viewmatrix,
	const int W, const int H,
	const float3 *dL_dnormals,
	const float4 *dL_dmean2Ds,
	float *dL_dTs,
	glm::vec3 *dL_dmeans,
	glm::vec3 *dL_dscales,
	glm::vec4 *dL_drots,
	const float VFOV_max,
	const float VFOV_min,
	const float HFOV_max,
	const float HFOV_min)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec3 scale;

	// Get transformation matrix of the Gaussian
	p_orig = p_origs[idx];
	rot = rots[idx];
	scale = scales[idx];
	R = quat_to_rotmat(rot);
	S = scale_to_mat(scale, 1.0f);

	glm::mat3 L = R * S;
	glm::mat3x4 M = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1));

	glm::mat4 world2camera = glm::mat4(
		viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[12],
		viewmatrix[1], viewmatrix[5], viewmatrix[9], viewmatrix[13],
		viewmatrix[2], viewmatrix[6], viewmatrix[10], viewmatrix[14],
		viewmatrix[3], viewmatrix[7], viewmatrix[11], viewmatrix[15]);

	glm::mat3x4 mat4x3_to_mat3 = glm::mat3x4(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);

	P = world2camera * mat4x3_to_mat3;
	T = glm::transpose(M) * P;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx * 9 + 0], dL_dTs[idx * 9 + 1], dL_dTs[idx * 9 + 2],
		dL_dTs[idx * 9 + 3], dL_dTs[idx * 9 + 4], dL_dTs[idx * 9 + 5],
		dL_dTs[idx * 9 + 6], dL_dTs[idx * 9 + 7], dL_dTs[idx * 9 + 8]);
	float4 dL_dmean2D = dL_dmean2Ds[idx];
	if (dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		const float Wrange = W / (HFOV_max - HFOV_min);
		const float Hrange = H / (VFOV_max - VFOV_min);

		const float u = T[0].z;
		const float v = T[1].z;
		const float w = T[2].z;
		const float r2_uw = u * u + w * w;
		const float r_uw = sqrt(u * u + w * w);
		const float r2 = u * u + v * v + w * w;
		const float r = sqrt(u * u + v * v + w * w);

		dL_dT[0].z += dL_dmean2D.x * Wrange * w / r2_uw - dL_dmean2D.y * Hrange * u * v / (r_uw * r2);
		dL_dT[1].z += dL_dmean2D.y * Hrange * r_uw / r2;
		dL_dT[2].z += -dL_dmean2D.x * Wrange * u / r2_uw - dL_dmean2D.y * Hrange * v * w / (r_uw * r2);
	}

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIABLE
	float multiplier = normal.z < 0 ? 1 : -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z));

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);

	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec3(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1]),
		0);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template <int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3 *means3D,
	const float *transMats,
	const int *radii,
	const float *shs,
	const bool *clamped,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	const float scale_modifier,
	const float *viewmatrix,
	const float *projmatrix,
	const float focal_x,
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3 *campos,
	// grad input
	float *dL_dtransMats,
	const float *dL_dnormal3Ds,
	float *dL_dcolors,
	float *dL_dshs,
	float4 *dL_dmean2Ds,
	glm::vec3 *dL_dmean3Ds,
	glm::vec3 *dL_dscales,
	glm::vec4 *dL_drots,
	const float VFOV_max,
	const float VFOV_min,
	const float HFOV_max,
	const float HFOV_min)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);

	compute_transmat_aabb(
		idx,
		means3D, scales, rotations,
		projmatrix, viewmatrix, W, H,
		(float3 *)dL_dnormal3Ds,
		dL_dmean2Ds,
		(dL_dtransMats),
		dL_dmean3Ds,
		dL_dscales,
		dL_drots,
		VFOV_max,
		VFOV_min,
		HFOV_max,
		HFOV_min);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3 *)means3D, *campos, shs, clamped, (glm::vec4 *)dL_dcolors, (glm::vec3 *)dL_dmean3Ds, (glm::vec4 *)dL_dshs);

	// hack the gradient here for densitification
	// float depth = transMats[idx * 9 + 8];
	// dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc
	// dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc

	const float u = transMats[idx * 9 + 2];
	const float v = transMats[idx * 9 + 5];
	const float w = transMats[idx * 9 + 8];
	const float dL_du = dL_dtransMats[idx * 9 + 2];
	const float dL_dv = dL_dtransMats[idx * 9 + 5];
	const float dL_dw = dL_dtransMats[idx * 9 + 8];

	// const float dx_du = w / (u * u + w * w);
	// const float dx_dw = -u / (u * u + w * w);
	// dL_dmean2Ds[idx].x = (dL_du / dx_du + dL_dw / dx_dw) * 0.5 * (HFOV_max - HFOV_min);

	// const float dy_du = -u * v / (sqrt(u * u + w * w) * (u * u + v * v + w * w));
	// const float dy_dv = sqrt(u * u + w * w) / (u * u + v * v + w * w);
	// const float dy_dw = -w * v / (sqrt(u * u + w * w) * (u * u + v * v + w * w));
	// dL_dmean2Ds[idx].y = (dL_du / dy_du + dL_dv / dy_dv + dL_dw / dy_dw) * 0.5 * (VFOV_max - VFOV_min) * W / H;

	const float phi = atan2f(u, w);
	// const float theta = atan2f(sqrt(u * u + w * w), -v);
	// const float r = sqrt(u * u + v * v + w * w);

	const float du_dphi = w;  // r * sin(theta) * cos(phi)
	const float dw_dphi = -u; // -r * sin(theta) * sin(phi)
	dL_dmean2Ds[idx].x = (dL_du * du_dphi + dL_dw * dw_dphi) * 0.5 * (HFOV_max - HFOV_min);

	const float du_dtheta = -v * sin(phi);		 // r * cos(theta) * sin(phi)
	const float dv_dtheta = sqrt(u * u + w * w); // r * sin(theta)
	const float dw_dtheta = -v * cos(phi);		 // r * cos(theta) * cos(phi)
	dL_dmean2Ds[idx].y = (dL_du * du_dtheta + dL_dv * dv_dtheta + dL_dw * dw_dtheta) * 0.5 * (VFOV_max - VFOV_min) * W / H;
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3 *means3D,
	const int *radii,
	const float *shs,
	const bool *clamped,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	const float scale_modifier,
	const float *transMats,
	const float *viewmatrix,
	const float *projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3 *campos,
	float4 *dL_dmean2D,
	float *dL_dnormals,
	float *dL_dtransMats,
	glm::vec3 *dL_dmean3D,
	float *dL_dcolor,
	float *dL_dsh,
	glm::vec3 *dL_dscale,
	glm::vec4 *dL_drot,
	const float vfov_min,
	const float vfov_max,
	const float hfov_min,
	const float hfov_max,
	const int width, int height)
{
	// vfov 角度制转弧度制
	const float VFOV_max = MY_PI / 2 - vfov_min * MY_PI / 180;
	const float VFOV_min = MY_PI / 2 - vfov_max * MY_PI / 180;

	// hfov 角度制转弧度制
	const float HFOV_max = hfov_max * MY_PI / 180;
	const float HFOV_min = hfov_min * MY_PI / 180;

	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		(float3 *)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec3 *)scales,
		(glm::vec4 *)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,
		dL_dtransMats,
		dL_dnormals,
		dL_dcolor,
		dL_dsh,
		dL_dmean2D,
		dL_dmean3D,
		dL_dscale,
		dL_drot,
		VFOV_max,
		VFOV_min,
		HFOV_max,
		HFOV_min);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	const int S, int W, int H,
	const float *bg_color,
	const float2 *means2D,
	const float4 *normal_opacity,
	const float *colors,
	const float *transMats,
	const float *depths,
	const float *features,
	const float *final_Ts,
	const int32_t *n_contrib,
	const float *dL_dpixels,
	const float *dL_depths,
	const float *dL_masks,
	const float *dL_dpix_feature,
	float *dL_dtransMat,
	float4 *dL_dmean2D,
	float *dL_dopacity,
	float *dL_dcolors,
	float *dL_dfeatures,
	float *dL_dnormals,
	const float vfov_min,
	const float vfov_max,
	const float hfov_min,
	const float hfov_max,
	const float scale_factor)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		S, W, H,
		bg_color,
		means2D,
		normal_opacity,
		transMats,
		colors,
		depths,
		features,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_depths,
		dL_masks,
		dL_dpix_feature,
		dL_dtransMat,
		dL_dmean2D,
		dL_dopacity,
		dL_dcolors,
		dL_dfeatures,
		dL_dnormals,
		vfov_min,
		vfov_max,
		hfov_min,
		hfov_max,
		scale_factor);
}