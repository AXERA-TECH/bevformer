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

// Denormalize bbox
void denormalize_bbox(const std::vector<float>& normalized_bbox,
                     const Config& config,
                     std::vector<float>& denormalized_bbox);

// Decode bboxes from model outputs
void decode_bboxes(const std::vector<std::vector<float>>& all_cls_scores,
                  const std::vector<std::vector<float>>& all_bbox_preds,
                  const Config& config,
                  std::vector<Detection3D>& detections);

// Apply score threshold filtering
void filter_by_score(const std::vector<Detection3D>& detections,
                    const Config& config,
                    float score_thr,
                    std::vector<Detection3D>& filtered);

// Circle NMS
void circle_nms(const std::vector<Detection3D>& detections,
               const Config& config,
               std::vector<int>& keep_indices);

// Post-process model outputs
DetectionResult post_process_outputs(
    const std::vector<std::vector<float>>& all_cls_scores,
    const std::vector<std::vector<float>>& all_bbox_preds,
    const Config& config,
    float score_thr = 0.1f);

} // namespace bevformer

