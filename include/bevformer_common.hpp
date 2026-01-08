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
#include <map>
#include <opencv2/opencv.hpp>
#include <ax_engine_api.h>

namespace bevformer {

extern const char* CLASS_NAMES[];
extern const int NUM_CLASSES;
extern const cv::Scalar CLASS_COLORS[];

struct Config {
    // Model parameters
    int bev_h = 200;
    int bev_w = 200;
    int embed_dims = 256;
    int num_cams = 6;
    
    // BBox coder
    std::vector<float> pc_range = {-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};
    std::vector<float> post_center_range = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};
    int max_num = 100;
    float score_threshold = 0.1f;
    
    // Image normalization
    std::vector<float> img_mean = {123.675f, 116.28f, 103.53f};
    std::vector<float> img_std = {58.395f, 57.12f, 57.375f};
    bool to_rgb = true;
    
    // Class score thresholds
    std::map<int, float> class_score_thrs = {
        {0, 0.3f}, {1, 0.3f}, {2, 0.3f}, {3, 0.3f}, {4, 0.3f},
        {5, 0.3f}, {6, 0.3f}, {7, 0.3f}, {8, 0.3f}, {9, 0.3f}
    };
    
    // Circle NMS distance thresholds
    std::map<int, float> dist_thrs = {
        {0, 2.0f}, {1, 3.0f}, {2, 2.5f}, {3, 4.0f}, {4, 3.0f},
        {5, 1.0f}, {6, 1.5f}, {7, 1.0f}, {8, 0.5f}, {9, 0.3f}
    };
};

struct Detection3D {
    std::vector<float> bbox;  // [x, y, z, w, l, h, yaw, vx, vy]
    float score;
    int label;
};

struct DetectionResult {
    std::vector<Detection3D> detections;
};

struct FrameInfo {
    std::vector<float> prev_bev;  // Previous BEV embedding
    std::string scene_token;
    std::vector<float> prev_pos;  // [x, y, z]
    float prev_angle;
};

} // namespace bevformer

