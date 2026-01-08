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
#include "bevformer_common.hpp"

namespace bevformer {

// Normalize image array
void normalize_image(const std::vector<float>& img,
                    const Config& img_norm_cfg,
                    std::vector<float>& normalized);

// Prepare CAN bus delta (for temporal modeling)
void compute_can_bus_delta(const std::vector<float>& curr_can_bus,
                          const std::vector<float>& prev_pos,
                          float prev_angle,
                          bool has_prev_bev,
                          std::vector<float>& delta_can_bus);

// Prepare previous BEV embedding
void prepare_prev_bev(const std::vector<float>& prev_bev,
                     int bev_h, int bev_w, int embed_dims,
                     std::vector<float>& prev_bev_input);

} // namespace bevformer

