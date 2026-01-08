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

#include "postprocess.hpp"
#include "utils.hpp"
#include "bevformer_common.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace bevformer {

void denormalize_bbox(const std::vector<float>& normalized_bbox,
                     const Config& config,
                     std::vector<float>& denormalized_bbox) {
    if (normalized_bbox.size() < 8) {
        return;
    }
    
    // Extract components 
    float cx = normalized_bbox[0];
    float cy = normalized_bbox[1];
    float cz = normalized_bbox[4];
    
    // Size: w, l, h (apply exp)
    float w = std::exp(normalized_bbox[2]);
    float l = std::exp(normalized_bbox[3]);
    float h = std::exp(normalized_bbox[5]);
    
    // Rotation: rot = atan2(rot_sine, rot_cosine)
    float rot_sine = normalized_bbox[6];
    float rot_cosine = normalized_bbox[7];
    float rot = std::atan2(rot_sine, rot_cosine);
    
    // Build output: [cx, cy, cz, w, l, h, rot, vx, vy] if velocity present
    // or [cx, cy, cz, w, l, h, rot] if not
    if (normalized_bbox.size() > 8) {
        // Velocity present
        denormalized_bbox.resize(9);
        denormalized_bbox[0] = cx;
        denormalized_bbox[1] = cy;
        denormalized_bbox[2] = cz;
        denormalized_bbox[3] = w;
        denormalized_bbox[4] = l;
        denormalized_bbox[5] = h;
        denormalized_bbox[6] = rot;
        denormalized_bbox[7] = normalized_bbox[8];  // vx
        denormalized_bbox[8] = normalized_bbox[9];  // vy
    } else {
        // No velocity
        denormalized_bbox.resize(7);
        denormalized_bbox[0] = cx;
        denormalized_bbox[1] = cy;
        denormalized_bbox[2] = cz;
        denormalized_bbox[3] = w;
        denormalized_bbox[4] = l;
        denormalized_bbox[5] = h;
        denormalized_bbox[6] = rot;
    }
}

void decode_bboxes(const std::vector<std::vector<float>>& all_cls_scores,
                  const std::vector<std::vector<float>>& all_bbox_preds,
                  const Config& config,
                  std::vector<Detection3D>& detections) {
    detections.clear();
    
    if (all_cls_scores.empty() || all_bbox_preds.empty()) {
        return;
    }
    
    const auto& cls_scores = all_cls_scores.back();  // Shape: (9000,) = (900 * 10)
    const auto& bbox_preds = all_bbox_preds.back();  // Shape: (9000,) = (900 * 10)
    
    size_t num_query = cls_scores.size() / NUM_CLASSES;  // Should be 900
    
    
    // Apply sigmoid to scores
    std::vector<float> scores_flat = sigmoid_vec(cls_scores);
    
    // TopK selection
    std::vector<std::pair<float, int>> score_idx;
    for (size_t i = 0; i < scores_flat.size(); ++i) {
        score_idx.push_back({scores_flat[i], i});
    }
    std::sort(score_idx.rbegin(), score_idx.rend());
    
    int max_num = std::min(config.max_num, (int)score_idx.size());
    
    
    // Collect topk indices 
    std::vector<int> topk_indices;
    std::vector<float> topk_scores;
    for (int i = 0; i < max_num; ++i) {
        topk_indices.push_back(score_idx[i].second);
        topk_scores.push_back(score_idx[i].first);
    }
    
    // Extract labels and bbox indices 
    std::vector<int> labels;
    std::vector<int> bbox_indices;
    for (int flat_idx : topk_indices) {
        labels.push_back(flat_idx % NUM_CLASSES);
        bbox_indices.push_back(flat_idx / NUM_CLASSES);
    }
    
    // Extract selected bboxes 
    std::vector<std::vector<float>> selected_bboxes;
    for (int query_idx : bbox_indices) {
        // Check bounds
        if (query_idx < 0 || query_idx >= (int)num_query || query_idx * 10 + 10 > (int)bbox_preds.size()) {
            // Out of bounds, use zero bbox
            selected_bboxes.push_back(std::vector<float>(10, 0.0f));
            continue;
        }
        std::vector<float> normalized_bbox(10);
        for (int j = 0; j < 10; ++j) {
            // bbox_preds layout: [query0_dim0, query0_dim1, ..., query0_dim9, query1_dim0, ...]
            // Index: query_idx * 10 + j
            normalized_bbox[j] = bbox_preds[query_idx * 10 + j];
        }
        selected_bboxes.push_back(normalized_bbox);
    }
    
    // Denormalize all bboxes at once 
    std::vector<std::vector<float>> final_box_preds;
    for (const auto& normalized_bbox : selected_bboxes) {
        std::vector<float> denormalized;
        denormalize_bbox(normalized_bbox, config, denormalized);
        final_box_preds.push_back(denormalized);
    }
    
    // Apply score threshold 
    std::vector<bool> thresh_mask(topk_scores.size(), true);
    if (config.score_threshold > 0) {
        for (size_t i = 0; i < topk_scores.size(); ++i) {
            thresh_mask[i] = (topk_scores[i] > config.score_threshold);
        }
        
        // If no detections pass threshold, try lowering it 
        bool any_passed = false;
        for (bool mask : thresh_mask) {
            if (mask) {
                any_passed = true;
                break;
            }
        }
        
        if (!any_passed) {
            float tmp_score = config.score_threshold;
            while (tmp_score >= 0.01f) {
                tmp_score *= 0.9f;
                for (size_t i = 0; i < topk_scores.size(); ++i) {
                    thresh_mask[i] = (topk_scores[i] >= tmp_score);
                }
                // Check if any passed now
                for (bool mask : thresh_mask) {
                    if (mask) {
                        any_passed = true;
                        break;
                    }
                }
                if (any_passed) break;
            }
            // If still no detections, accept all 
            if (!any_passed) {
                std::fill(thresh_mask.begin(), thresh_mask.end(), true);
            }
        }
    }
    
    // Apply post_center_range filtering 
    std::vector<bool> range_mask = thresh_mask;  // Start with thresh_mask
    if (config.post_center_range.size() >= 6) {
        for (size_t i = 0; i < final_box_preds.size(); ++i) {
            if (!thresh_mask[i]) {
                range_mask[i] = false;
                continue;
            }
            const auto& bbox = final_box_preds[i];
            if (bbox.size() < 3) {
                range_mask[i] = false;
                continue;
            }
            float x = bbox[0];
            float y = bbox[1];
            float z = bbox[2];
            
            bool in_range = (x >= config.post_center_range[0] && x <= config.post_center_range[3] &&
                            y >= config.post_center_range[1] && y <= config.post_center_range[4] &&
                            z >= config.post_center_range[2] && z <= config.post_center_range[5]);
            range_mask[i] = in_range;
        }
    }
    
    // Collect final detections 
    for (size_t i = 0; i < final_box_preds.size(); ++i) {
        if (!range_mask[i]) {
            continue;
        }
        
        Detection3D det;
        det.bbox = final_box_preds[i];
        det.score = topk_scores[i];
        det.label = labels[i];
        
        detections.push_back(det);
    }
    
}

void filter_by_score(const std::vector<Detection3D>& detections,
                    const Config& config,
                    float score_thr,
                    std::vector<Detection3D>& filtered) {
    filtered.clear();
    
    
    for (const auto& det : detections) {
        float thr = config.class_score_thrs.count(det.label) > 0 ?
                   config.class_score_thrs.at(det.label) : score_thr;
        
        if (det.score > thr) {
            filtered.push_back(det);
        }
    }
    
}

void circle_nms(const std::vector<Detection3D>& detections,
               const Config& config,
               std::vector<int>& keep_indices) {
    keep_indices.clear();
    
    if (detections.empty()) {
        return;
    }
    
    // Sort by score (descending)
    std::vector<std::pair<float, int>> score_idx;
    for (size_t i = 0; i < detections.size(); ++i) {
        score_idx.push_back({detections[i].score, i});
    }
    std::sort(score_idx.rbegin(), score_idx.rend());
    
    // Create sorted arrays 
    std::vector<Detection3D> sorted_detections;
    std::vector<int> order;
    for (const auto& pair : score_idx) {
        sorted_detections.push_back(detections[pair.second]);
        order.push_back(pair.second);
    }
    
    std::vector<bool> suppressed(sorted_detections.size(), false);
    
    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        keep_indices.push_back(order[i]);
        
        const auto& det_i = sorted_detections[i];
        float radius = config.dist_thrs.count(det_i.label) > 0 ?
                      config.dist_thrs.at(det_i.label) : 1.0f;
        
        if (i + 1 < sorted_detections.size()) {
            float x_i = det_i.bbox[0];
            float y_i = det_i.bbox[1];
            
            for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
                if (suppressed[j]) continue;
                
                const auto& det_j = sorted_detections[j];
                // Check same class
                if (det_i.label != det_j.label) continue;
                
                float dx = det_j.bbox[0] - x_i;
                float dy = det_j.bbox[1] - y_i;
                float dist = std::sqrt(dx * dx + dy * dy);
                
                if (dist < radius) {
                    suppressed[j] = true;
                }
            }
        }
    }
}

DetectionResult post_process_outputs(
    const std::vector<std::vector<float>>& all_cls_scores,
    const std::vector<std::vector<float>>& all_bbox_preds,
    const Config& config,
    float score_thr) {
    
    DetectionResult result;
    
    // Decode bboxes 
    std::vector<Detection3D> detections;
    decode_bboxes(all_cls_scores, all_bbox_preds, config, detections);
    
    // Adjust z coordinate and shrink dimensions 
    for (auto& det : detections) {
        if (det.bbox.size() >= 6) {
            // Adjust z coordinate: convert center z to bottom center z
            det.bbox[2] = det.bbox[2] - det.bbox[5] * 0.5f;
            // Shrink box dimensions: multiply w, l, h by 0.9
            det.bbox[3] *= 0.9f;  // w
            det.bbox[4] *= 0.9f;  // l
            det.bbox[5] *= 0.9f;  // h
        }
    }
    
    
    // Apply class score thresholds 
    std::vector<Detection3D> filtered;
    filter_by_score(detections, config, score_thr, filtered);
    
    
    if (filtered.empty()) {
        return result;  // Return empty result
    }
    
    // Apply Circle NMS 
    std::vector<int> keep_indices;
    circle_nms(filtered, config, keep_indices);
    
    
    if (keep_indices.empty()) {
        return result;  // Return empty result
    }
    
    // Final detections
    for (int idx : keep_indices) {
        result.detections.push_back(filtered[idx]);
    }
    
    return result;
}

} // namespace bevformer

