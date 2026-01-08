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

// Load JSON configuration file
bool load_config_from_json(const std::string& config_path, Config& config);

// Load scene index
bool load_scene_index(const std::string& data_dir, 
                      std::map<std::string, std::vector<int>>& scenes);

// File utilities
bool file_exists(const std::string& path);
std::string join_path(const std::string& dir, const std::string& filename);

// Math utilities
float sigmoid(float x);
std::vector<float> sigmoid_vec(const std::vector<float>& x);

// Array utilities
template<typename T>
void print_array_shape(const std::vector<T>& arr, const std::string& name);

// Progress bar utilities
void print_progress_bar(size_t current, size_t total, const std::string& prefix = "", 
                        float elapsed_ms = 0.0f);

} // namespace bevformer

