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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
	auto lambda = [&t](size_t N)
	{
		t.resize_({(long long)N});
		return reinterpret_cast<char *>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &colors,
	const torch::Tensor &features,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &mask,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const bool prefiltered,
	const bool debug,
	const float vfov_min,
	const float vfov_max,
	const float hfov_min,
	const float hfov_max,
	const float scale_factor)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int S = features.size(1);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_contrib = torch::full({2, H, W}, 0.0, int_opts);
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_feature = torch::full({S + 3, H, W}, 0.0, float_opts);  // feature(S) + normal(3)
	torch::Tensor out_depth = torch::full({4, H, W}, 0.0, float_opts);
	torch::Tensor out_T = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, S, degree, M,
			background.contiguous().data_ptr<float>(),
			W, H,
			means3D.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data_ptr<float>(),
			features.contiguous().data_ptr<float>(),
			opacity.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data_ptr<float>(),
			mask.contiguous().data_ptr<bool>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_contrib.contiguous().data_ptr<int>(),
			out_color.contiguous().data_ptr<float>(),
			out_feature.contiguous().data_ptr<float>(),
			out_depth.contiguous().data_ptr<float>(),
			out_T.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			debug,
			vfov_min,
			vfov_max,
			hfov_min,
			hfov_max,
			scale_factor);
	}
	return std::make_tuple(rendered, out_contrib, out_color, out_feature, out_depth, out_T, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	const torch::Tensor &colors,
	const torch::Tensor &features,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &dL_dout_color,
	const torch::Tensor &dL_dout_depth,
	const torch::Tensor &dL_dout_mask,
	const torch::Tensor &dL_dout_feature,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const torch::Tensor &geomBuffer,
	const int R,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const torch::Tensor &out_contrib,
	const bool debug,
	const float vfov_min,
	const float vfov_max,
	const float hfov_min,
	const float hfov_max,
	const float scale_factor)
{
	const int P = means3D.size(0);
	const int S = features.size(1);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if (sh.size(0) != 0)
	{
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dfeatures = torch::zeros({P, S}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

	torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
	torch::Tensor dL_dnormals = torch::zeros({P, 3}, means3D.options());

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::backward(P, S, degree, M, R,
											 background.contiguous().data_ptr<float>(),
											 W, H,
											 means3D.contiguous().data_ptr<float>(),
											 sh.contiguous().data_ptr<float>(),
											 colors.contiguous().data_ptr<float>(),
											 features.contiguous().data_ptr<float>(),
											 scales.data_ptr<float>(),
											 scale_modifier,
											 rotations.data_ptr<float>(),
											 cov3D_precomp.contiguous().data_ptr<float>(),
											 viewmatrix.contiguous().data_ptr<float>(),
											 projmatrix.contiguous().data_ptr<float>(),
											 campos.contiguous().data_ptr<float>(),
											 tan_fovx,
											 tan_fovy,
											 radii.contiguous().data_ptr<int>(),
											 reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
											 out_contrib.contiguous().data_ptr<int>(),
											 dL_dout_color.contiguous().data_ptr<float>(),
											 dL_dout_depth.contiguous().data_ptr<float>(),
											 dL_dout_mask.contiguous().data_ptr<float>(),
											 dL_dout_feature.contiguous().data_ptr<float>(),
											 dL_dmeans2D.contiguous().data_ptr<float>(),
											 dL_dopacity.contiguous().data_ptr<float>(),
											 dL_dcolors.contiguous().data_ptr<float>(),
											 dL_dmeans3D.contiguous().data_ptr<float>(),
											 dL_dcov3D.contiguous().data_ptr<float>(),
											 dL_dsh.contiguous().data_ptr<float>(),
											 dL_dfeatures.contiguous().data_ptr<float>(),
											 dL_dscales.contiguous().data_ptr<float>(),
											 dL_drotations.contiguous().data_ptr<float>(),
											 dL_dtransMat.contiguous().data_ptr<float>(),
											 dL_dnormals.contiguous().data_ptr<float>(),
											 debug,
											 vfov_min,
											 vfov_max,
											 hfov_min,
											 hfov_max,
											 scale_factor);
	}

	return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dfeatures, dL_dopacity, dL_dmeans3D, dL_dcov3D,
						   dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
	torch::Tensor &means3D,
	torch::Tensor &viewmatrix,
	torch::Tensor &projmatrix)
{
	const int P = means3D.size(0);

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(P,
												means3D.contiguous().data_ptr<float>(),
												viewmatrix.contiguous().data_ptr<float>(),
												projmatrix.contiguous().data_ptr<float>(),
												present.contiguous().data_ptr<bool>());
	}

	return present;
}