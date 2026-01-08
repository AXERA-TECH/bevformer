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
#include <string>
#include <opencv2/opencv.hpp>
#include "bevformer_common.hpp"

namespace bevformer {

struct MetaData {
    std::vector<std::vector<float>> lidar2img;  // (N, 4, 4)
    std::vector<float> can_bus;  // (18,)
    std::vector<int> img_shape;  // [H, W, C]
    int num_cams = 6;
    Config img_norm_cfg;
};

// Load metadata from JSON
bool load_meta(const std::string& meta_path, MetaData& meta);

// Preprocess single image
cv::Mat preprocess_image(const std::string& img_path, 
                        const Config& img_norm_cfg,
                        const std::vector<int>& target_size);

// Load data for a frame
bool load_frame_data(const std::string& data_dir,
                    const std::string& scene_name,
                    int frame_idx,
                    std::vector<cv::Mat>& images,  // Output: (N, C, H, W)
                    std::vector<std::vector<float>>& lidar2img,  // Output: (N, 4, 4)
                    std::vector<float>& can_bus,  // Output: (18,)
                    MetaData& meta);

// Convert OpenCV Mat to model input format
void prepare_model_input(const std::vector<cv::Mat>& images,
                        std::vector<float>& img_tensor);  // Output: (1, N, C, H, W)

} // namespace bevformer

