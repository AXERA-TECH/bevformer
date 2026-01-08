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

#include "data_loader.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cstdio>
#include <omp.h>

namespace bevformer {

// Simple JSON value extractor
float extract_float(const std::string& json_str, const std::string& key) {
    size_t pos = json_str.find("\"" + key + "\"");
    if (pos == std::string::npos) return 0.0f;
    pos = json_str.find(":", pos);
    if (pos == std::string::npos) return 0.0f;
    pos = json_str.find_first_of("-0123456789", pos);
    if (pos == std::string::npos) return 0.0f;
    size_t end = json_str.find_first_not_of("0123456789.-eE", pos);
    if (end == std::string::npos) end = json_str.length();
    return std::stof(json_str.substr(pos, end - pos));
}

std::vector<float> extract_array(const std::string& json_str, const std::string& key) {
    std::vector<float> result;
    size_t pos = json_str.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    pos = json_str.find("[", pos);
    if (pos == std::string::npos) return result;
    size_t end = json_str.find("]", pos);
    if (end == std::string::npos) return result;
    
    std::string arr_str = json_str.substr(pos + 1, end - pos - 1);
    std::istringstream iss(arr_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\n"));
        token.erase(token.find_last_not_of(" \t\n") + 1);
        try {
            result.push_back(std::stof(token));
        } catch (...) {
            // Skip invalid numbers
        }
    }
    return result;
}

bool load_meta(const std::string& meta_path, MetaData& meta) {
    std::ifstream file(meta_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open meta file: " << meta_path << std::endl;
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    
    // Simplified parsing
    // Extract lidar2img: nested array (N, 4, 4) where N is number of cameras
    meta.lidar2img.clear();
    
    // Parse nested array: "lidar2img": [[[1,0,0,0],[0,1,0,0],...], ...]
    size_t lidar2img_pos = content.find("\"lidar2img\"");
    if (lidar2img_pos != std::string::npos) {
        size_t arr_start = content.find("[", lidar2img_pos);
        if (arr_start != std::string::npos) {
            // Parse nested array properly by extracting all numbers and tracking bracket levels
            int bracket_level = 0;
            std::vector<float> current_matrix;
            std::string current_number;
            bool in_number = false;
            
            for (size_t i = arr_start; i < content.length() && i < arr_start + 20000; ++i) {
                char c = content[i];
                
                if (c == '[') {
                    bracket_level++;
                    if (bracket_level == 2) {
                        // Start of a 4x4 matrix (camera matrix)
                        current_matrix.clear();
                    }
                } else if (c == ']') {
                    // Flush any pending number
                    if (in_number && !current_number.empty()) {
                        try {
                            current_matrix.push_back(std::stof(current_number));
                        } catch (...) {}
                        current_number.clear();
                        in_number = false;
                    }
                    
                    if (bracket_level == 2) {
                        // End of a 4x4 matrix
                        if (current_matrix.size() == 16) {
                            meta.lidar2img.push_back(current_matrix);
                        }
                        current_matrix.clear();
                    } else if (bracket_level == 1) {
                        // End of lidar2img array
                        bracket_level--;
                        break;
                    }
                    bracket_level--;
                } else if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    // Delimiter or whitespace - flush number if we were parsing one
                    if (in_number && !current_number.empty()) {
                        try {
                            current_matrix.push_back(std::stof(current_number));
                        } catch (...) {}
                        current_number.clear();
                        in_number = false;
                    }
                } else if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                    // Part of a number
                    current_number += c;
                    in_number = true;
                }
            }
        }
    }
    
    // Fallback: if still empty, use identity matrices
    if (meta.lidar2img.empty()) {
        int num_cams = meta.num_cams > 0 ? meta.num_cams : 6;
        for (int cam = 0; cam < num_cams; ++cam) {
            std::vector<float> matrix(16, 0.0f);
            matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;  // Identity matrix
            meta.lidar2img.push_back(matrix);
        }
    }
    
    // Extract can_bus
    meta.can_bus = extract_array(content, "can_bus");
    if (meta.can_bus.size() < 18) {
        meta.can_bus.resize(18, 0.0f);
    }
    
    // Extract img_shape
    std::vector<float> img_shape_float = extract_array(content, "img_shape");
    meta.img_shape.clear();
    for (float val : img_shape_float) {
        meta.img_shape.push_back((int)val);
    }
    if (meta.img_shape.size() < 3) {
        meta.img_shape = {480, 800, 3};
    }
    
    // Extract num_cams
    meta.num_cams = (int)extract_float(content, "num_cams");
    if (meta.num_cams <= 0) meta.num_cams = 6;
    
    // Extract img_norm_cfg
    meta.img_norm_cfg.img_mean = extract_array(content, "mean");
    if (meta.img_norm_cfg.img_mean.size() < 3) {
        meta.img_norm_cfg.img_mean = {123.675f, 116.28f, 103.53f};
    }
    meta.img_norm_cfg.img_std = extract_array(content, "std");
    if (meta.img_norm_cfg.img_std.size() < 3) {
        meta.img_norm_cfg.img_std = {58.395f, 57.12f, 57.375f};
    }
    // Extract to_rgb (default: true)
    size_t to_rgb_pos = content.find("\"to_rgb\"");
    if (to_rgb_pos != std::string::npos) {
        size_t val_start = content.find_first_not_of(" \t\n:", to_rgb_pos + 8);
        if (val_start != std::string::npos) {
            std::string val_str = content.substr(val_start, 5);  // "true" or "false"
            meta.img_norm_cfg.to_rgb = (val_str.find("true") == 0);
        } else {
            meta.img_norm_cfg.to_rgb = true;  // Default
        }
    } else {
        meta.img_norm_cfg.to_rgb = true;  // Default 
    }
    
    // Warning only shown once (using static variable)
    static bool warning_shown = false;
    if (!warning_shown) {
        std::cerr << "Warning: Using simplified JSON parser. Consider using jsoncpp library." << std::endl;
        warning_shown = true;
    }
    
    return true;
}

cv::Mat preprocess_image(const std::string& img_path,
                        const Config& img_norm_cfg,
                        const std::vector<int>& target_size) {
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << img_path << std::endl;
        return cv::Mat();
    }
    
    if (img_norm_cfg.to_rgb) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    
    cv::Mat resized;
    if (target_size.size() >= 2) {
        int target_h = target_size[0];
        int target_w = target_size[1];
        if (img.rows != target_h || img.cols != target_w) {
            cv::resize(img, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img;
        }
    } else {
        resized = img;
    }
    
    resized.convertTo(resized, CV_32F);
    
    std::vector<cv::Mat> channels;
    cv::split(resized, channels);
    if (channels.size() >= 3 && img_norm_cfg.img_mean.size() >= 3 && img_norm_cfg.img_std.size() >= 3) {
        for (size_t c = 0; c < 3; ++c) {
            channels[c] = (channels[c] - img_norm_cfg.img_mean[c]) / img_norm_cfg.img_std[c];
        }
    }
    cv::merge(channels, resized);
    
    return resized;
}

bool load_frame_data(const std::string& data_dir,
                    const std::string& scene_name,
                    int frame_idx,
                    std::vector<cv::Mat>& images,
                    std::vector<std::vector<float>>& lidar2img,
                    std::vector<float>& can_bus,
                    MetaData& meta) {
    std::string scene_dir = join_path(data_dir, scene_name);
    
    char meta_filename[64];
    snprintf(meta_filename, sizeof(meta_filename), "meta_%06d.json", frame_idx);
    std::string meta_path = join_path(scene_dir, meta_filename);
    
    if (!load_meta(meta_path, meta)) {
        return false;
    }
    
    std::vector<int> target_size;
    if (meta.img_shape.size() >= 3) {
        target_size = {meta.img_shape[0], meta.img_shape[1]};
    } else {
        target_size = {480, 800};
    }
    
    images.resize(meta.num_cams);
    bool load_success = true;
    
    #pragma omp parallel for num_threads(std::min(meta.num_cams, 6))
    for (int cam_idx = 0; cam_idx < meta.num_cams; ++cam_idx) {
        char img_filename[64];
        snprintf(img_filename, sizeof(img_filename), "cam_%02d_%06d.png", cam_idx, frame_idx);
        std::string img_path = join_path(scene_dir, img_filename);
        cv::Mat img = preprocess_image(img_path, meta.img_norm_cfg, target_size);
        if (img.empty()) {
            #pragma omp critical
            {
                std::cerr << "Failed to load camera " << cam_idx << " image: " << img_path << std::endl;
                load_success = false;
            }
        } else {
            images[cam_idx] = img;
        }
    }
    
    if (!load_success) {
        return false;
    }
    
    lidar2img = meta.lidar2img;
    can_bus = meta.can_bus;
    
    return true;
}

void prepare_model_input(const std::vector<cv::Mat>& images,
                        std::vector<float>& img_tensor) {
    if (images.empty()) {
        return;
    }
    
    int num_cams = images.size();
    int C = images[0].channels();
    int H = images[0].rows;
    int W = images[0].cols;
    
    // Allocate tensor: (1, N, C, H, W)
    img_tensor.resize(1 * num_cams * C * H * W);
    
    // Layout: (1, N, C, H, W) - channel-first format
    size_t plane_size = H * W;
    size_t cam_size = C * plane_size;
    
    #pragma omp parallel for
    for (int n = 0; n < num_cams; ++n) {
        const cv::Mat& img = images[n];
        float* cam_ptr = img_tensor.data() + n * cam_size;
        
        std::vector<cv::Mat> channels(C);
        for (int c = 0; c < C; ++c) {
            channels[c] = cv::Mat(H, W, CV_32F, cam_ptr + c * plane_size);
        }
        cv::split(img, channels);
    }
}

} // namespace bevformer

