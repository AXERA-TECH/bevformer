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

#include "bevformer_common.hpp"

namespace bevformer {

const char* CLASS_NAMES[] = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
};

const int NUM_CLASSES = 10;

const cv::Scalar CLASS_COLORS[] = {
    cv::Scalar(0, 255, 0),      // car - green
    cv::Scalar(255, 255, 0),    // truck - cyan
    cv::Scalar(0, 0, 255),      // construction_vehicle - red
    cv::Scalar(0, 165, 255),    // bus - orange
    cv::Scalar(255, 0, 255),    // trailer - magenta
    cv::Scalar(0, 255, 255),    // barrier - yellow
    cv::Scalar(128, 0, 128),    // motorcycle - purple
    cv::Scalar(255, 165, 0),    // bicycle - blue
    cv::Scalar(0, 0, 255),      // pedestrian - red
    cv::Scalar(128, 128, 128)   // traffic_cone - gray
};

} // namespace bevformer

