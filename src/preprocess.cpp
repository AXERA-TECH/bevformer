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

#include "preprocess.hpp"
#include "bevformer_common.hpp"
#include <cmath>

namespace bevformer {

void normalize_image(const std::vector<float>& img,
                    const Config& img_norm_cfg,
                    std::vector<float>& normalized) {
    normalized.resize(img.size());
    for (size_t i = 0; i < img.size(); ++i) {
        int c = (i / (img.size() / 3)) % 3;
        normalized[i] = (img[i] - img_norm_cfg.img_mean[c]) / img_norm_cfg.img_std[c];
    }
}

void compute_can_bus_delta(const std::vector<float>& curr_can_bus,
                          const std::vector<float>& prev_pos,
                          float prev_angle,
                          bool has_prev_bev,
                          std::vector<float>& delta_can_bus) {
    delta_can_bus = curr_can_bus;
    
    if (has_prev_bev && prev_pos.size() >= 3) {
        delta_can_bus[0] -= prev_pos[0];
        delta_can_bus[1] -= prev_pos[1];
        delta_can_bus[2] -= prev_pos[2];
    } else {
        delta_can_bus[0] = 0.0f;
        delta_can_bus[1] = 0.0f;
        delta_can_bus[2] = 0.0f;
    }
    
    if (has_prev_bev) {
        if (delta_can_bus.size() > 17) {
            delta_can_bus[17] -= prev_angle;
        } else if (delta_can_bus.size() > 0) {
            delta_can_bus.back() -= prev_angle;
        }
    } else {
        if (delta_can_bus.size() > 17) {
            delta_can_bus[17] = 0.0f;
        } else if (delta_can_bus.size() > 0) {
            delta_can_bus.back() = 0.0f;
        }
    }
}

void prepare_prev_bev(const std::vector<float>& prev_bev,
                     int bev_h, int bev_w, int embed_dims,
                     std::vector<float>& prev_bev_input) {
    size_t expected_size = static_cast<size_t>(bev_h) * static_cast<size_t>(bev_w) * static_cast<size_t>(embed_dims);
    
    if (prev_bev.empty()) {
        prev_bev_input.resize(expected_size, 0.0f);
    } else {
        prev_bev_input = prev_bev;
        if (prev_bev_input.size() != expected_size) {
            prev_bev_input.resize(expected_size, 0.0f);
        }
    }
}

} // namespace bevformer

