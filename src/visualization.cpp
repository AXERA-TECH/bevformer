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

#include "visualization.hpp"
#include "bevformer_common.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <omp.h>

namespace bevformer {

cv::Mat denormalize_image(const std::vector<float>& img_array,
                         const Config& img_norm_cfg) {
    // This function is for denormalizing model input images
    // For visualization, we typically use the original camera images
    // This is a placeholder - actual implementation depends on data format
    cv::Mat img(480, 800, CV_8UC3, cv::Scalar(0, 0, 0));
    return img;
}

std::vector<cv::Point3f> compute_bbox_corners(const Detection3D& bbox) {
    // Extract bbox parameters
    float x = bbox.bbox[0];  // center x
    float y = bbox.bbox[1];  // center y
    float z = bbox.bbox[2];  // bottom center z
    float w = bbox.bbox[3];  // width (y direction)
    float l = bbox.bbox[4];  // length (x direction)
    float h = bbox.bbox[5];  // height (z direction)
    float yaw = bbox.bbox[6];
    
    yaw = yaw - (M_PI / 2.0f - M_PI / 18.0f);
    
    int order[] = {0, 1, 3, 2, 4, 5, 7, 6};
    float corners_norm_raw[8][3] = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
        {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
    };
    
    float corners_norm[8][3];
    for (int i = 0; i < 8; ++i) {
        int idx = order[i];
        corners_norm[i][0] = corners_norm_raw[idx][0] - 0.5f;  // x: -0.5 to 0.5
        corners_norm[i][1] = corners_norm_raw[idx][1] - 0.5f;  // y: -0.5 to 0.5
        corners_norm[i][2] = corners_norm_raw[idx][2] - 0.0f;  // z: 0 to 1 (bottom to top)
    }
    
    float corners_local[8][3];
    for (int i = 0; i < 8; ++i) {
        corners_local[i][0] = l * corners_norm[i][0];  // x_size (length)
        corners_local[i][1] = w * corners_norm[i][1];  // y_size (width)
        corners_local[i][2] = h * corners_norm[i][2];  // z_size (height)
    }
    
    float cos_yaw = std::cos(yaw);
    float sin_yaw = std::sin(yaw);
    
    std::vector<cv::Point3f> corners(8);
    for (int i = 0; i < 8; ++i) {
        float lx = corners_local[i][0];
        float ly = corners_local[i][1];
        float lz = corners_local[i][2];
        
        float rx = lx * cos_yaw + ly * sin_yaw;
        float ry = -lx * sin_yaw + ly * cos_yaw;
        float rz = lz;  // z unchanged
        
        corners[i] = cv::Point3f(x + rx, y + ry, z + rz);
    }
    
    return corners;
}

cv::Mat draw_bbox3d_on_img(const Detection3D& bbox,
                          const cv::Mat& img,
                          const std::vector<float>& lidar2img,
                          const cv::Scalar& color,
                          int thickness) {
    cv::Mat result = img.clone();
    
    if (lidar2img.size() != 16) {
        return result;
    }
    
    // Compute corners
    std::vector<cv::Point3f> corners_3d = compute_bbox_corners(bbox);
    
    cv::Mat lidar2img_mat(4, 4, CV_32F);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            lidar2img_mat.at<float>(r, c) = lidar2img[r * 4 + c];
        }
    }
    
    std::vector<cv::Point2f> corners_2d;
    for (const auto& pt3d : corners_3d) {
        cv::Mat pt4d = (cv::Mat_<float>(4, 1) << pt3d.x, pt3d.y, pt3d.z, 1.0f);
        cv::Mat pt2d = lidar2img_mat * pt4d;  // (4, 1)
        
        float w = std::max(1e-5f, std::min(1e5f, pt2d.at<float>(2)));
        
        float u = pt2d.at<float>(0) / w;
        float v = pt2d.at<float>(1) / w;
        corners_2d.push_back(cv::Point2f(u, v));
    }
    
    // Draw lines
    int line_indices[][2] = {
        {0, 1}, {0, 3}, {0, 4}, {1, 2}, {1, 5}, {3, 2}, {3, 7},
        {4, 5}, {4, 7}, {2, 6}, {5, 6}, {6, 7}
    };
    
    int h = img.rows;
    int w = img.cols;
    
    for (int i = 0; i < 12; ++i) {
        cv::Point2f pt1 = corners_2d[line_indices[i][0]];
        cv::Point2f pt2 = corners_2d[line_indices[i][1]];
        
        bool pt1_in = (0 <= pt1.x && pt1.x < w && 0 <= pt1.y && pt1.y < h);
        bool pt2_in = (0 <= pt2.x && pt2.x < w && 0 <= pt2.y && pt2.y < h);
        
        if (pt1_in || pt2_in) {
            cv::line(result, pt1, pt2, color, thickness, cv::LINE_AA);
        }
    }
    
    return result;
}

cv::Mat draw_bev_map(const std::vector<Detection3D>& detections,
                    const Config& config,
                    float score_thr,
                    int bev_size) {
    int bev_w = bev_size;
    int bev_h = bev_size;
    cv::Mat bev_img(bev_h, bev_w, CV_8UC3, cv::Scalar(255, 255, 255));  // White background
    
    if (config.pc_range.size() < 6) {
        return bev_img;
    }
    
    float x_min = config.pc_range[0];
    float y_min = config.pc_range[1];
    float x_max = config.pc_range[3];
    float y_max = config.pc_range[4];
    
    float x_range = x_max - x_min;
    float y_range = y_max - y_min;
    
    cv::Scalar grid_color(200, 200, 200);
    for (int i = -5; i <= 5; ++i) {
        float x = x_min + (i + 5) * x_range / 10.0f;
        float y = y_min + (i + 5) * y_range / 10.0f;
        
        int img_x = (int)((y - y_min) / y_range * bev_w);
        if (0 <= img_x && img_x < bev_w) {
            cv::line(bev_img, cv::Point(img_x, 0), cv::Point(img_x, bev_h), grid_color, 1);
        }
        int img_y = (int)((x_max - x) / x_range * bev_h);
        if (0 <= img_y && img_y < bev_h) {
            cv::line(bev_img, cv::Point(0, img_y), cv::Point(bev_w, img_y), grid_color, 1);
        }
    }
    
    int center_x = (int)((0 - y_min) / y_range * bev_w);
    int center_y = (int)((x_max - 0) / x_range * bev_h);
    cv::line(bev_img, cv::Point(center_x, 0), cv::Point(center_x, bev_h), cv::Scalar(150, 150, 150), 2);
    cv::line(bev_img, cv::Point(0, center_y), cv::Point(bev_w, center_y), cv::Scalar(150, 150, 150), 2);
    
    int ego_length_px = 30;  // pixels (representing ~4.5m, along x-axis rightward)
    int ego_width_px = 12;   // pixels (representing ~1.8m, along y-axis downward)
    
    float rotation_angle_90 = M_PI / 2.0f;  // 90 degrees in radians
    float cos_rot_90 = std::cos(rotation_angle_90);
    float sin_rot_90 = std::sin(rotation_angle_90);
    
    std::vector<cv::Point2f> ego_corners_local = {
        cv::Point2f(ego_length_px/2.0f, -ego_width_px/2.0f),   // front-top (head)
        cv::Point2f(ego_length_px/2.0f, ego_width_px/2.0f),    // front-bottom
        cv::Point2f(-ego_length_px/2.0f, ego_width_px/2.0f),   // back-bottom
        cv::Point2f(-ego_length_px/2.0f, -ego_width_px/2.0f)   // back-top
    };
    
    // Rotate twice (90 degrees each)
    std::vector<cv::Point2f> ego_corners_rotated;
    for (const auto& corner : ego_corners_local) {
        float rx = corner.x * cos_rot_90 - corner.y * sin_rot_90;
        float ry = corner.x * sin_rot_90 + corner.y * cos_rot_90;
        float rx2 = rx * cos_rot_90 - ry * sin_rot_90;
        float ry2 = rx * sin_rot_90 + ry * cos_rot_90;
        ego_corners_rotated.push_back(cv::Point2f(rx2, ry2));
    }
    
    // Translate to image coordinates
    std::vector<cv::Point> ego_corners;
    for (const auto& corner : ego_corners_rotated) {
        int corner_img_x = (int)(center_x + corner.x);
        int corner_img_y = (int)(center_y + corner.y);
        ego_corners.push_back(cv::Point(corner_img_x, corner_img_y));
    }
    
    // Draw filled rectangle
    cv::fillPoly(bev_img, std::vector<std::vector<cv::Point>>{ego_corners}, cv::Scalar(0, 0, 255));  // Red filled
    cv::polylines(bev_img, std::vector<std::vector<cv::Point>>{ego_corners}, true, cv::Scalar(0, 0, 0), 2);  // Black outline
    
    // Draw arrow
    int arrow_length = ego_length_px / 2;
    cv::Point2f initial_direction(1.0f, 0.0f);
    float arrow_dir_x = initial_direction.x * cos_rot_90 - initial_direction.y * sin_rot_90;
    float arrow_dir_y = initial_direction.x * sin_rot_90 + initial_direction.y * cos_rot_90;
    float arrow_dir_x2 = arrow_dir_x * cos_rot_90 - arrow_dir_y * sin_rot_90;
    float arrow_dir_y2 = arrow_dir_x * sin_rot_90 + arrow_dir_y * cos_rot_90;
    int arrow_end_x = (int)(center_x + arrow_length * arrow_dir_x2);
    int arrow_end_y = (int)(center_y + arrow_length * arrow_dir_y2);
    cv::arrowedLine(bev_img, cv::Point(center_x, center_y), cv::Point(arrow_end_x, arrow_end_y),
                   cv::Scalar(0, 0, 0), 3, cv::LINE_AA, 0, 0.3);  // Black arrow
    
    std::vector<Detection3D> filtered_detections;
    if (score_thr > 0) {
        for (const auto& det : detections) {
            if (det.score > score_thr) {
                filtered_detections.push_back(det);
            }
        }
    } else {
        filtered_detections = detections;
    }
    
    if (filtered_detections.empty()) {
        // Rotate and flip before returning
        cv::Point2f center(bev_w / 2.0f, bev_h / 2.0f);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, 90, 1.0);
        cv::warpAffine(bev_img, bev_img, rot_mat, cv::Size(bev_w, bev_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        cv::flip(bev_img, bev_img, 1);
        
        // Add text
        std::string text = "BEV Map";
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 1.0;
        int thickness = 2;
        cv::Size text_size = cv::getTextSize(text, font, font_scale, thickness, nullptr);
        int text_x = bev_w - text_size.width - 10;
        int text_y = text_size.height + 10;
        cv::putText(bev_img, text, cv::Point(text_x, text_y), font, font_scale, cv::Scalar(0, 0, 0), thickness);
        
        return bev_img;
    }
    
    cv::Scalar default_color(255, 255, 255);
    for (const auto& det : filtered_detections) {
        if (det.bbox.size() < 7) continue;
        
        float x = det.bbox[0];
        float y = det.bbox[1];
        // z (det.bbox[2]) and h (det.bbox[5]) not used in BEV visualization
        float w = det.bbox[3];
        float l = det.bbox[4];
        float yaw = det.bbox[6];
        
        yaw = yaw - M_PI / 2.0f;
        
        int img_x = (int)((y - y_min) / y_range * bev_w);
        int img_y = (int)((x_max - x) / x_range * bev_h);
        
        if (!(0 <= img_x && img_x < bev_w && 0 <= img_y && img_y < bev_h)) {
            continue;
        }
        
        cv::Scalar color = CLASS_COLORS[det.label % NUM_CLASSES];
        
        int box_l_px = (int)(l / y_range * bev_h);
        
        float cos_yaw = std::cos(yaw);
        float sin_yaw = std::sin(yaw);
        
        std::vector<cv::Point2f> corners_local = {
            cv::Point2f(l/2.0f, w/2.0f),   // front-right
            cv::Point2f(l/2.0f, -w/2.0f),  // front-left
            cv::Point2f(-l/2.0f, -w/2.0f), // back-left
            cv::Point2f(-l/2.0f, w/2.0f)   // back-right
        };
        
        // Rotate corners
        std::vector<cv::Point2f> corners_rotated;
        for (const auto& corner : corners_local) {
            float rx = corner.x * cos_yaw - corner.y * sin_yaw;
            float ry = corner.x * sin_yaw + corner.y * cos_yaw;
            corners_rotated.push_back(cv::Point2f(rx, ry));
        }
        
        // Translate to world coordinates and convert to image space
        std::vector<cv::Point> corners_img;
        for (const auto& corner : corners_rotated) {
            float corner_x = x + corner.x;  // x in LiDAR (forward)
            float corner_y = y + corner.y;  // y in LiDAR (left)
            int corner_img_x = (int)((corner_y - y_min) / y_range * bev_w);  // y -> img_x
            int corner_img_y = (int)((x_max - corner_x) / x_range * bev_h);  // x -> img_y (flipped)
            corners_img.push_back(cv::Point(corner_img_x, corner_img_y));
        }
        
        cv::Mat overlay = bev_img.clone();
        cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{corners_img}, color);
        cv::addWeighted(overlay, 0.5, bev_img, 0.5, 0, bev_img);
        // Draw outline
        cv::polylines(bev_img, std::vector<std::vector<cv::Point>>{corners_img}, true, cv::Scalar(0, 0, 0), 2);
        
        int arrow_length = std::max(box_l_px / 2, 10);
        int arrow_end_x = (int)(img_x + arrow_length * sin_yaw);   // y component -> img_x
        int arrow_end_y = (int)(img_y - arrow_length * cos_yaw);  // x component -> img_y (flipped)
        cv::arrowedLine(bev_img, cv::Point(img_x, img_y), cv::Point(arrow_end_x, arrow_end_y),
                       cv::Scalar(0, 0, 0), 2, cv::LINE_AA, 0, 0.3);  // Black arrow
        
        // Draw center point
        cv::circle(bev_img, cv::Point(img_x, img_y), 3, cv::Scalar(0, 0, 0), -1);  // Black center point
    }
    
    cv::Point2f center(bev_w / 2.0f, bev_h / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, 90, 1.0);
    cv::warpAffine(bev_img, bev_img, rot_mat, cv::Size(bev_w, bev_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    
    cv::flip(bev_img, bev_img, 1);
    
    std::string text = "BEV Map";
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    int thickness = 2;
    cv::Size text_size = cv::getTextSize(text, font, font_scale, thickness, nullptr);
    int text_x = bev_w - text_size.width - 10;
    int text_y = text_size.height + 10;
    cv::putText(bev_img, text, cv::Point(text_x, text_y), font, font_scale, cv::Scalar(0, 0, 0), thickness);
    
    return bev_img;
}

cv::Mat visualize_results(const std::vector<cv::Mat>& camera_imgs,
                         const DetectionResult& result,
                         const std::vector<std::vector<float>>& lidar2img,
                         const Config& config,
                         float score_thr) {
    // Filter detections first (before any image processing)
    std::vector<Detection3D> filtered_detections;
    if (score_thr > 0 && !result.detections.empty()) {
        filtered_detections.reserve(result.detections.size());
        for (const auto& det : result.detections) {
            if (det.score > score_thr) {
                filtered_detections.push_back(det);
            }
        }
    } else {
        filtered_detections = result.detections;
    }
    
    // Convert normalized images back to BGR for visualization (optimized)
    std::vector<cv::Mat> vis_imgs(camera_imgs.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < camera_imgs.size(); ++i) {
        const cv::Mat& img = camera_imgs[i];
        cv::Mat vis_img;
        if (img.type() == CV_32FC3) {
            // Denormalize in-place without extra clone
            cv::Mat denorm;
            std::vector<cv::Mat> channels;
            cv::split(img, channels);
            if (channels.size() >= 3 && config.img_mean.size() >= 3 && config.img_std.size() >= 3) {
                for (size_t c = 0; c < 3; ++c) {
                    channels[c] = channels[c] * config.img_std[c] + config.img_mean[c];
                }
            }
            cv::merge(channels, denorm);
            denorm.convertTo(vis_img, CV_8UC3);
            if (config.to_rgb) {
                cv::cvtColor(vis_img, vis_img, cv::COLOR_RGB2BGR);
            }
        } else {
            vis_img = img.clone();
        }
        vis_imgs[i] = vis_img;
    }
    
    // Draw 3D bboxes on camera images (in parallel)
    if (!filtered_detections.empty() && lidar2img.size() == vis_imgs.size()) {
        #pragma omp parallel for
        for (size_t cam_idx = 0; cam_idx < vis_imgs.size(); ++cam_idx) {
            if (cam_idx >= lidar2img.size()) continue;
            for (const auto& det : filtered_detections) {
                cv::Scalar color = CLASS_COLORS[det.label % NUM_CLASSES];
                try {
                    vis_imgs[cam_idx] = draw_bbox3d_on_img(det, vis_imgs[cam_idx], lidar2img[cam_idx], color, 2);
                } catch (...) {
                    // Skip if drawing fails
                }
            }
        }
    }
    
    // Draw BEV map
    int bev_size = vis_imgs.empty() ? 800 : vis_imgs[0].rows;
    cv::Mat bev_img;
    if (!filtered_detections.empty()) {
        bev_img = draw_bev_map(filtered_detections, config, score_thr, bev_size);
    } else {
        bev_img = cv::Mat(bev_size, bev_size, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::putText(bev_img, "BEV Map (No Detections)", cv::Point(10, bev_size/2),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
    }
    
    // Combine images (optimized - avoid unnecessary clones)
    if (vis_imgs.size() == 6) {
        int target_height = vis_imgs[0].rows;
        for (size_t i = 1; i < vis_imgs.size(); ++i) {
            target_height = std::max(target_height, vis_imgs[i].rows);
        }
        
        // Resize only if needed, reuse memory
        std::vector<cv::Mat> resized_imgs(6);
        for (size_t i = 0; i < 6; ++i) {
            if (vis_imgs[i].rows == target_height) {
                resized_imgs[i] = vis_imgs[i];  // No copy, just reference
            } else {
                cv::resize(vis_imgs[i], resized_imgs[i], 
                          cv::Size(vis_imgs[i].cols * target_height / vis_imgs[i].rows, target_height));
            }
        }
        
        // Reorder and flip rear cameras
        std::vector<cv::Mat> reordered(6);
        reordered[0] = resized_imgs[2];
        reordered[1] = resized_imgs[0];
        reordered[2] = resized_imgs[1];
        cv::flip(resized_imgs[4], reordered[3], 1);
        cv::flip(resized_imgs[3], reordered[4], 1);
        cv::flip(resized_imgs[5], reordered[5], 1);
        
        // Concatenate rows
        cv::Mat top_row, bottom_row;
        cv::hconcat(std::vector<cv::Mat>(reordered.begin(), reordered.begin() + 3), top_row);
        cv::hconcat(std::vector<cv::Mat>(reordered.begin() + 3, reordered.end()), bottom_row);
        
        cv::Mat left_side;
        cv::vconcat(top_row, bottom_row, left_side);
        
        // Resize BEV if needed
        if (bev_img.rows != left_side.rows) {
            int new_w = bev_img.cols * left_side.rows / bev_img.rows;
            cv::resize(bev_img, bev_img, cv::Size(new_w, left_side.rows));
        }
        
        cv::Mat final;
        cv::hconcat(left_side, bev_img, final);
        return final;
    } else if (vis_imgs.size() > 1) {
        // Python: resized_imgs = [img if img.shape[0] == target_height else cv2.resize(...) for img in vis_imgs]
        int target_height = 0;
        for (const auto& img : vis_imgs) {
            target_height = std::max(target_height, img.rows);
        }
        
        std::vector<cv::Mat> resized_imgs;
        for (const auto& img : vis_imgs) {
            if (img.rows == target_height) {
                resized_imgs.push_back(img.clone());
            } else {
                cv::Mat resized;
                cv::resize(img, resized, cv::Size(img.cols * target_height / img.rows, target_height));
                resized_imgs.push_back(resized);
            }
        }
        
        // Python: vis_img = np.hstack([np.hstack(resized_imgs), bev_img])
        cv::Mat camera_combined;
        cv::hconcat(resized_imgs, camera_combined);
        
        if (bev_img.rows != target_height) {
            int new_w = bev_img.cols * target_height / bev_img.rows;
            cv::resize(bev_img, bev_img, cv::Size(new_w, target_height));
        }
        
        cv::Mat final;
        cv::hconcat(camera_combined, bev_img, final);
        return final;
    } else {
        // Python: Single camera case
        cv::Mat cam_img = vis_imgs.empty() ? bev_img : vis_imgs[0];
        if (bev_img.rows != cam_img.rows) {
            int new_w = bev_img.cols * cam_img.rows / bev_img.rows;
            cv::resize(bev_img, bev_img, cv::Size(new_w, cam_img.rows));
        }
        cv::Mat final;
        if (!vis_imgs.empty()) {
            cv::hconcat(cam_img, bev_img, final);
        } else {
            final = bev_img;
        }
        return final;
    }
}

bool create_video_from_images(const std::string& image_dir,
                             const std::string& output_video_path,
                             std::string& actual_video_path,
                             int fps) {
    actual_video_path = output_video_path;  // Start with requested path
    // Build glob patterns with proper path separator
    std::string png_pattern = image_dir;
    if (png_pattern.back() != '/') {
        png_pattern += "/";
    }
    png_pattern += "*.png";
    
    std::vector<cv::String> image_files;
    cv::glob(png_pattern, image_files);
    
    // Also try jpg/jpeg (append to same vector)
    std::string jpg_pattern = image_dir;
    if (jpg_pattern.back() != '/') {
        jpg_pattern += "/";
    }
    jpg_pattern += "*.jpg";
    std::vector<cv::String> jpg_files;
    cv::glob(jpg_pattern, jpg_files);
    image_files.insert(image_files.end(), jpg_files.begin(), jpg_files.end());
    
    std::string jpeg_pattern = image_dir;
    if (jpeg_pattern.back() != '/') {
        jpeg_pattern += "/";
    }
    jpeg_pattern += "*.jpeg";
    std::vector<cv::String> jpeg_files;
    cv::glob(jpeg_pattern, jpeg_files);
    image_files.insert(image_files.end(), jpeg_files.begin(), jpeg_files.end());
    
    if (image_files.empty()) {
        std::cerr << "Warning: No image files found in " << image_dir << " for video creation" << std::endl;
        std::cerr << "  Searched patterns: " << png_pattern << ", " << jpg_pattern << ", " << jpeg_pattern << std::endl;
        return false;
    }
    
    std::sort(image_files.begin(), image_files.end());
    
    cv::Mat first_img = cv::imread(image_files[0]);
    if (first_img.empty()) {
        std::cerr << "Warning: Failed to read first image: " << image_files[0] << std::endl;
        return false;
    }
    
    int width = first_img.cols;
    int height = first_img.rows;
    
    // Limit size
    int max_width = 1920, max_height = 1080;
    if (width > max_width || height > max_height) {
        float scale = std::min((float)max_width / width, (float)max_height / height);
        width = (int)(width * scale);
        height = (int)(height * scale);
    }
    
    // Try multiple codecs in order of preference
    cv::VideoWriter writer;
    bool opened = false;
    std::string used_codec;
    
    // Try H264 first (if available)
    writer = cv::VideoWriter(actual_video_path, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));
    if (writer.isOpened()) {
        opened = true;
        used_codec = "H264";
    } else {
        // Try mp4v
        writer = cv::VideoWriter(actual_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
        if (writer.isOpened()) {
            opened = true;
            used_codec = "mp4v";
        } else {
            // Try XVID
            writer = cv::VideoWriter(actual_video_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(width, height));
            if (writer.isOpened()) {
                opened = true;
                used_codec = "XVID";
            } else {
                // Try MJPG (Motion JPEG, usually well supported)
                writer = cv::VideoWriter(actual_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
                if (writer.isOpened()) {
                    opened = true;
                    used_codec = "MJPG";
                } else {
                    // Try AVI format with different codec
                    std::string avi_path = actual_video_path;
                    if (avi_path.rfind(".mp4") == avi_path.length() - 4) {
                        avi_path.replace(avi_path.length() - 4, 4, ".avi");
                    }
                    writer = cv::VideoWriter(avi_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
                    if (writer.isOpened()) {
                        opened = true;
                        used_codec = "MJPG (AVI)";
                        actual_video_path = avi_path;  // Update output path
                    }
                }
            }
        }
    }
    
    if (!opened) {
        std::cerr << "Warning: Failed to create video writer for " << actual_video_path << std::endl;
        std::cerr << "  Tried codecs: H264, mp4v, XVID, MJPG" << std::endl;
        std::cerr << "  Video size: " << width << "x" << height << ", FPS: " << fps << std::endl;
        std::cerr << "  Note: OpenCV may not have video encoding support. Consider using ffmpeg to create video from images." << std::endl;
        return false;
    }
    
    int frame_count = 0;
    for (const auto& img_path : image_files) {
        cv::Mat img = cv::imread(img_path);
        if (!img.empty()) {
            if (img.cols != width || img.rows != height) {
                cv::resize(img, img, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            }
            writer.write(img);
            frame_count++;
        }
    }
    
    writer.release();
    
    if (frame_count == 0) {
        std::cerr << "Warning: No valid images written to video" << std::endl;
        return false;
    }
    
    std::cout << "Created video with " << frame_count << " frames using codec " << used_codec 
              << ": " << actual_video_path << std::endl;
    return true;
}

} // namespace bevformer

