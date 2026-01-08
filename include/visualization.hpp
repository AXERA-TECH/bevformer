/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 *
 * Copyright (c) 2025, AXERA Semiconductor Co., Ltd. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

/*
 * Author: GUOFANGMING
 */

#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "bevformer_common.hpp"

namespace bevformer {

// Denormalize image for visualization
cv::Mat denormalize_image(const std::vector<float>& img_array,
                         const Config& img_norm_cfg);

// Compute 3D bbox corners
std::vector<cv::Point3f> compute_bbox_corners(const Detection3D& bbox);

// Draw 3D bbox on image
cv::Mat draw_bbox3d_on_img(const Detection3D& bbox,
                          const cv::Mat& img,
                          const std::vector<float>& lidar2img,
                          const cv::Scalar& color = cv::Scalar(0, 255, 0),
                          int thickness = 2);

// Draw BEV map
cv::Mat draw_bev_map(const std::vector<Detection3D>& detections,
                    const Config& config,
                    float score_thr = 0.1f,
                    int bev_size = 800);

// Visualize results (cameras + BEV)
cv::Mat visualize_results(const std::vector<cv::Mat>& camera_imgs,
                         const DetectionResult& result,
                         const std::vector<std::vector<float>>& lidar2img,
                         const Config& config,
                         float score_thr = 0.1f);

// Create video from images
// Returns the actual video path created (may differ from output_video_path if codec fails)
bool create_video_from_images(const std::string& image_dir,
                             const std::string& output_video_path,
                             std::string& actual_video_path,
                             int fps = 3);

} // namespace bevformer

