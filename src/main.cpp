
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

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include "bevformer_common.hpp"
#include "data_loader.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"
#include "visualization.hpp"
#include "utils.hpp"
#include "timer.hpp"

// Middleware for IO handling
namespace middleware {
    void prepare_io(AX_ENGINE_IO_INFO_T* info, AX_ENGINE_IO_T* io_data) {
        memset(io_data, 0, sizeof(AX_ENGINE_IO_T));
        
        io_data->nInputSize = info->nInputSize;
        io_data->nOutputSize = info->nOutputSize;
        io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
        io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
        
        memset(io_data->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);
        memset(io_data->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
        
        for (uint32_t i = 0; i < info->nInputSize; ++i) {
            auto& meta = info->pInputs[i];
            auto& buffer = io_data->pInputs[i];
            buffer.nSize = meta.nSize;
            int ret = AX_SYS_MemAlloc((AX_U64*)(&buffer.phyAddr), &buffer.pVirAddr, meta.nSize, 128, (const AX_S8*)"bevformer");
            if (ret != 0) {
                std::cerr << "Failed to allocate input buffer " << i << std::endl;
            }
        }
        
        for (uint32_t i = 0; i < info->nOutputSize; ++i) {
            auto& meta = info->pOutputs[i];
            auto& buffer = io_data->pOutputs[i];
            buffer.nSize = meta.nSize;
            int ret = AX_SYS_MemAlloc((AX_U64*)(&buffer.phyAddr), &buffer.pVirAddr, meta.nSize, 128, (const AX_S8*)"bevformer");
            if (ret != 0) {
                std::cerr << "Failed to allocate output buffer " << i << std::endl;
            }
        }
    }
    
    void free_io(AX_ENGINE_IO_T* io) {
        if (!io) return;
        
        for (uint32_t i = 0; i < io->nInputSize; ++i) {
            if (io->pInputs[i].phyAddr != 0) {
                AX_SYS_MemFree(io->pInputs[i].phyAddr, io->pInputs[i].pVirAddr);
            }
        }
        for (uint32_t i = 0; i < io->nOutputSize; ++i) {
            if (io->pOutputs[i].phyAddr != 0) {
                AX_SYS_MemFree(io->pOutputs[i].phyAddr, io->pOutputs[i].pVirAddr);
            }
        }
        delete[] io->pInputs;
        delete[] io->pOutputs;
    }
}

struct Args {
    std::string model_path;
    std::string config_json;
    std::string data_dir;
    std::string output_dir = "./inference_results_extracted";
    float score_thr = 0.1f;
    int fps = 3;
    int start_scene = 0;
    int end_scene = -1;
};

bool parse_args(int argc, char* argv[], Args& args) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.axmodel> <config_json> <data_dir> [options]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  <model.axmodel>        Path to AX model file (*.axmodel)" << std::endl;
        std::cerr << "  <config_json>          Path to model config JSON file" << std::endl;
        std::cerr << "  <data_dir>             Path to extracted data directory" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --output-dir <dir>     Output directory (default: ./inference_results_extracted)" << std::endl;
        std::cerr << "  --score-thr <float>    Score threshold (default: 0.1)" << std::endl;
        std::cerr << "  --fps <int>            Video FPS (default: 3)" << std::endl;
        std::cerr << "  --start-scene <int>    Start scene index (default: 0)" << std::endl;
        std::cerr << "  --end-scene <int>      End scene index (default: all)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " bevformer.axmodel model_config.json ./extracted_data --output-dir ./results" << std::endl;
        return false;
    }
    
    args.model_path = argv[1];
    args.config_json = argv[2];
    args.data_dir = argv[3];
    
    if (args.model_path.find(".axmodel") == std::string::npos && 
        args.model_path.find(".joint") == std::string::npos) {
        std::cerr << "Warning: Model file '" << args.model_path << "' does not have .axmodel or .joint extension." << std::endl;
        std::cerr << "         First argument should be the path to the AX model file." << std::endl;
    }
    
    for (int i = 4; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        
        std::string key = argv[i];
        std::string value = argv[i + 1];
        
        if (key == "--output-dir") {
            args.output_dir = value;
        } else if (key == "--score-thr") {
            args.score_thr = std::stof(value);
        } else if (key == "--fps") {
            args.fps = std::stoi(value);
        } else if (key == "--start-scene") {
            args.start_scene = std::stoi(value);
        } else if (key == "--end-scene") {
            args.end_scene = std::stoi(value);
        }
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return -1;
    }
    
    // Load configuration
    bevformer::Config config;
    if (!bevformer::load_config_from_json(args.config_json, config)) {
        std::cerr << "Failed to load config" << std::endl;
        return -1;
    }
    
    // Load scene index
    std::map<std::string, std::vector<int>> scenes;
    if (!bevformer::load_scene_index(args.data_dir, scenes)) {
        std::cerr << "Failed to load scene index from: " << args.data_dir << std::endl;
        std::cerr << "Please ensure scene_index.json exists or scene directories are present" << std::endl;
        return -1;
    }
    
    if (scenes.empty()) {
        std::cerr << "No scenes found in data directory: " << args.data_dir << std::endl;
        std::cerr << "Please check:" << std::endl;
        std::cerr << "  1. scene_index.json exists in " << args.data_dir << std::endl;
        std::cerr << "  2. Or scene_* directories exist in " << args.data_dir << std::endl;
        return -1;
    }
    
    std::cout << "Total scenes to process: " << scenes.size() << std::endl;
    for (const auto& pair : scenes) {
        std::cout << "  - " << pair.first << ": " << pair.second.size() << " frames" << std::endl;
    }
    
    // Create output directory
    system(("mkdir -p " + args.output_dir).c_str());
    
    // Initialize AX Engine
    AX_SYS_Init();
    
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    int ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) {
        std::cerr << "AX_ENGINE_Init failed: " << ret << std::endl;
        return -1;
    }
    
    // Load model
    std::ifstream model_file(args.model_path, std::ios::binary);
    if (!model_file) {
        std::cerr << "Failed to open model file: " << args.model_path << std::endl;
        return -1;
    }
    
    model_file.seekg(0, std::ios::end);
    size_t model_size = model_file.tellg();
    model_file.seekg(0, std::ios::beg);
    
    std::vector<char> model_buffer(model_size);
    model_file.read(model_buffer.data(), model_size);
    model_file.close();
    
    // Create handle
    AX_ENGINE_HANDLE handle;
    ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), model_buffer.size());
    if (ret != 0) {
        std::cerr << "AX_ENGINE_CreateHandle failed with error code: " << ret << std::endl;
        std::cerr << std::endl;
        std::cerr << "Possible causes:" << std::endl;
        std::cerr << "  1. Model file is not a valid .axmodel file" << std::endl;
        std::cerr << "  2. Model file is corrupted or incompatible" << std::endl;
        std::cerr << "  3. Wrong argument order - first argument should be the model file" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Provided model path: " << args.model_path << std::endl;
        std::cerr << "Model file size: " << model_size << " bytes" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Correct usage:" << std::endl;
        std::cerr << "  " << "bevformer_inference <model.axmodel> <config.json> <data_dir>" << std::endl;
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }
    
    // Create context
    ret = AX_ENGINE_CreateContext(handle);
    if (ret != 0) {
        std::cerr << "AX_ENGINE_CreateContext failed: " << ret << std::endl;
        AX_ENGINE_DestroyHandle(handle);
        AX_ENGINE_Deinit();
        return -1;
    }
    
    // Get IO info
    AX_ENGINE_IO_INFO_T* io_info;
    ret = AX_ENGINE_GetIOInfo(handle, &io_info);
    if (ret != 0) {
        std::cerr << "AX_ENGINE_GetIOInfo failed: " << ret << std::endl;
        AX_ENGINE_DestroyHandle(handle);
        AX_ENGINE_Deinit();
        return -1;
    }
    
    // Allocate IO
    AX_ENGINE_IO_T io_data;
    middleware::prepare_io(io_info, &io_data);
    
    // Process scenes
    std::vector<std::string> scene_names;
    for (const auto& pair : scenes) {
        scene_names.push_back(pair.first);
    }
    
    int end_scene = (args.end_scene < 0) ? scene_names.size() : std::min(args.end_scene, (int)scene_names.size());
    
    bevformer::FrameInfo frame_info;
    frame_info.prev_pos = {0.0f, 0.0f, 0.0f};
    frame_info.prev_angle = 0.0f;
    
    for (int scene_idx = args.start_scene; scene_idx < end_scene; ++scene_idx) {
        std::string scene_name = scene_names[scene_idx];
        std::vector<int> sample_indices = scenes[scene_name];
        
        std::cout << "Processing scene " << (scene_idx + 1) << "/" << scene_names.size()
                  << ": " << scene_name << " (" << sample_indices.size() << " frames)" << std::endl;
        
        // Reset for new scene
        if (frame_info.scene_token != scene_name) {
            frame_info.prev_bev.clear();
            frame_info.prev_pos = {0.0f, 0.0f, 0.0f};
            frame_info.prev_angle = 0.0f;
        }
        frame_info.scene_token = scene_name;
        
        // Create scene output directory
        std::string scene_dir = bevformer::join_path(args.output_dir, scene_name);
        system(("mkdir -p " + scene_dir).c_str());
        std::string images_dir = bevformer::join_path(scene_dir, "images");
        system(("mkdir -p " + images_dir).c_str());
        
        // Process frames
        std::vector<float> load_times;
        std::vector<float> inference_times;
        std::vector<float> preprocess_times;
        std::vector<float> postprocess_times;
        std::vector<float> visualize_times;
        std::vector<float> total_times;
        
        bevformer::Timer scene_timer;
        scene_timer.start();
        
        for (size_t local_idx = 0; local_idx < sample_indices.size(); ++local_idx) {
            int frame_idx = sample_indices[local_idx];
            
            bevformer::Timer frame_timer;
            bevformer::Timer step_timer;
            
            // Load data
            step_timer.start();
            std::vector<cv::Mat> images;
            std::vector<std::vector<float>> lidar2img;
            std::vector<float> can_bus;
            bevformer::MetaData meta;
            
            if (!bevformer::load_frame_data(args.data_dir, scene_name, frame_idx,
                                           images, lidar2img, can_bus, meta)) {
                std::cerr << "Failed to load frame data" << std::endl;
                continue;
            }
            float load_time = step_timer.cost_ms();
            load_times.push_back(load_time);
            
            // Prepare model inputs
            step_timer.reset();
            std::vector<float> img_tensor;
            bevformer::prepare_model_input(images, img_tensor);
            
            // Save current values first (before computing delta)
            std::vector<float> tmp_pos = {can_bus[0], can_bus[1], can_bus[2]};
            float tmp_angle = (can_bus.size() > 17) ? can_bus[17] : 0.0f;
            
            // Check if we have prev_bev and are in the same scene
            bool has_prev_bev = !frame_info.prev_bev.empty() && 
                               (frame_info.scene_token == scene_name);
            
            // Compute delta
            std::vector<float> delta_can_bus;
            bevformer::compute_can_bus_delta(can_bus, frame_info.prev_pos,
                                            frame_info.prev_angle, has_prev_bev, delta_can_bus);
            
            // Update prev_pos and prev_angle AFTER computing delta
            frame_info.prev_pos = tmp_pos;
            frame_info.prev_angle = tmp_angle;
            
            // Prepare prev_bev
            std::vector<float> prev_bev_input;
            bevformer::prepare_prev_bev(frame_info.prev_bev, config.bev_h, config.bev_w,
                                       config.embed_dims, prev_bev_input);
            
            // Copy inputs to engine buffers
            // Find input indices by name
            int img_input_idx = -1, can_bus_input_idx = -1;
            int lidar2img_input_idx = -1, prev_bev_input_idx = -1;
            
            for (uint32_t i = 0; i < io_info->nInputSize; ++i) {
                const char* name = io_info->pInputs[i].pName;
                if (strcmp(name, "img") == 0) img_input_idx = i;
                else if (strcmp(name, "can_bus") == 0) can_bus_input_idx = i;
                else if (strcmp(name, "lidar2img") == 0) lidar2img_input_idx = i;
                else if (strcmp(name, "prev_bev") == 0) prev_bev_input_idx = i;
            }
            
            // Copy data
            if (img_input_idx >= 0) {
                memcpy(io_data.pInputs[img_input_idx].pVirAddr, img_tensor.data(),
                      std::min(img_tensor.size() * sizeof(float), (size_t)io_data.pInputs[img_input_idx].nSize));
            }
            
            if (can_bus_input_idx >= 0) {
                memcpy(io_data.pInputs[can_bus_input_idx].pVirAddr, delta_can_bus.data(),
                      std::min(delta_can_bus.size() * sizeof(float), (size_t)io_data.pInputs[can_bus_input_idx].nSize));
            }
            
            if (lidar2img_input_idx >= 0) {
                std::vector<float> lidar2img_flat;
                for (const auto& mat : lidar2img) {
                    lidar2img_flat.insert(lidar2img_flat.end(), mat.begin(), mat.end());
                }
                memcpy(io_data.pInputs[lidar2img_input_idx].pVirAddr, lidar2img_flat.data(),
                      std::min(lidar2img_flat.size() * sizeof(float), (size_t)io_data.pInputs[lidar2img_input_idx].nSize));
            }
            
            if (prev_bev_input_idx >= 0) {
                memcpy(io_data.pInputs[prev_bev_input_idx].pVirAddr, prev_bev_input.data(),
                      std::min(prev_bev_input.size() * sizeof(float), (size_t)io_data.pInputs[prev_bev_input_idx].nSize));
            }
            float preprocess_time = step_timer.cost_ms();
            preprocess_times.push_back(preprocess_time);
            
            // Run inference
            step_timer.reset();
            ret = AX_ENGINE_RunSync(handle, &io_data);
            float inference_time = step_timer.cost_ms();
            inference_times.push_back(inference_time);
            
            if (ret != 0) {
                std::cerr << "AX_ENGINE_RunSync failed: " << ret << std::endl;
                continue;
            }
            
            // Get outputs
            std::vector<std::vector<float>> all_cls_scores;
            std::vector<std::vector<float>> all_bbox_preds;
            std::vector<float> bev_embed;
            
            
            for (uint32_t i = 0; i < io_info->nOutputSize; ++i) {
                const char* name = io_info->pOutputs[i].pName;
                float* data = (float*)io_data.pOutputs[i].pVirAddr;
                size_t size = io_data.pOutputs[i].nSize / sizeof(float);
                
                std::string name_str(name);
                
                // Check for outputs_classes (cls scores)
                // Shape can be: [6, 1, 900, 10] (all layers) or [1, 900, 10] (last layer only)
                if (name_str == "outputs_classes" || name_str.find("classes") != std::string::npos) {
                    if (io_info->pOutputs[i].pShape != nullptr) {
                        int ndim = io_info->pOutputs[i].nShapeSize;
                        if (ndim >= 4) {
                            // Shape: [num_layers, batch, num_query, num_classes]
                            int num_layers = io_info->pOutputs[i].pShape[0];
                            int batch = io_info->pOutputs[i].pShape[1];
                            int num_query = io_info->pOutputs[i].pShape[2];
                            int num_classes = io_info->pOutputs[i].pShape[3];
                            
                            size_t layer_size = batch * num_query * num_classes;
                            for (int layer = 0; layer < num_layers; ++layer) {
                                std::vector<float> layer_data(data + layer * layer_size, 
                                                              data + (layer + 1) * layer_size);
                                all_cls_scores.push_back(layer_data);
                            }
                        } else if (ndim == 3) {
                            // Shape: [batch, num_query, num_classes] - single layer (last decoder layer)
                            // Python: all_cls_scores[-1] gets shape (bs, num_query, num_classes)
                            all_cls_scores.push_back(std::vector<float>(data, data + size));
                        } else {
                            // Unknown shape, treat as flattened single layer
                            all_cls_scores.push_back(std::vector<float>(data, data + size));
                        }
                    } else {
                        // No shape info, treat as single layer
                        all_cls_scores.push_back(std::vector<float>(data, data + size));
                    }
                }
                // Check for outputs_coords (bbox predictions)
                // Shape can be: [6, 1, 900, 10] (all layers) or [1, 900, 10] (last layer only)
                else if (name_str == "outputs_coords" || name_str.find("coords") != std::string::npos) {
                    if (io_info->pOutputs[i].pShape != nullptr) {
                        int ndim = io_info->pOutputs[i].nShapeSize;
                        if (ndim >= 4) {
                            // Shape: [num_layers, batch, num_query, bbox_dims]
                            int num_layers = io_info->pOutputs[i].pShape[0];
                            int batch = io_info->pOutputs[i].pShape[1];
                            int num_query = io_info->pOutputs[i].pShape[2];
                            int bbox_dims = io_info->pOutputs[i].pShape[3];
                            
                            size_t layer_size = batch * num_query * bbox_dims;
                            for (int layer = 0; layer < num_layers; ++layer) {
                                std::vector<float> layer_data(data + layer * layer_size, 
                                                              data + (layer + 1) * layer_size);
                                all_bbox_preds.push_back(layer_data);
                            }
                        } else if (ndim == 3) {
                            // Shape: [batch, num_query, bbox_dims] - single layer (last decoder layer)
                            all_bbox_preds.push_back(std::vector<float>(data, data + size));
                        } else {
                            // Unknown shape, treat as flattened single layer
                            all_bbox_preds.push_back(std::vector<float>(data, data + size));
                        }
                    } else {
                        // No shape info, treat as single layer
                        all_bbox_preds.push_back(std::vector<float>(data, data + size));
                    }
                }
                // Legacy names (for compatibility)
                else if (name_str.find("cls") != std::string::npos || name_str.find("score") != std::string::npos) {
                    all_cls_scores.push_back(std::vector<float>(data, data + size));
                } else if (name_str.find("bbox") != std::string::npos || name_str.find("pred") != std::string::npos) {
                    all_bbox_preds.push_back(std::vector<float>(data, data + size));
                } else if (name_str.find("bev") != std::string::npos) {
                    bev_embed = std::vector<float>(data, data + size);
                }
            }
            
            
            // Update prev_bev
            frame_info.prev_bev = bev_embed;
            
            // Post-process
            step_timer.reset();  // reset() already calls start()
            bevformer::DetectionResult result = bevformer::post_process_outputs(
                all_cls_scores, all_bbox_preds, config, args.score_thr);
            float postprocess_time = step_timer.cost_ms();  // cost_ms() automatically stops if running
            postprocess_times.push_back(postprocess_time);
            
            
            // Visualize
            step_timer.reset();
            cv::Mat vis_img = bevformer::visualize_results(images, result, lidar2img, config, args.score_thr);
            float visualize_time = step_timer.cost_ms();
            visualize_times.push_back(visualize_time);
            
            // Save visualization (use 6-digit frame index to match Python version)
            if (!vis_img.empty()) {
                char frame_filename[64];
                snprintf(frame_filename, sizeof(frame_filename), "frame_%06zu.png", local_idx);
                std::string output_path = bevformer::join_path(images_dir, frame_filename);
                if (!cv::imwrite(output_path, vis_img)) {
                    std::cerr << "Warning: Failed to save image: " << output_path << std::endl;
                }
            } else {
                std::cerr << "Warning: Empty visualization image for frame " << local_idx << std::endl;
            }
            
            float total_time = frame_timer.cost_ms();
            total_times.push_back(total_time);
            
            // Show progress bar with elapsed time
            float elapsed_ms = scene_timer.cost_ms();
            bevformer::print_progress_bar(local_idx + 1, sample_indices.size(), 
                                         "Processing:", elapsed_ms);
        }
        
        // Print statistics
        if (!inference_times.empty()) {
            float avg_load = load_times.empty() ? 0.0f : std::accumulate(load_times.begin(), load_times.end(), 0.0f) / load_times.size();
            float avg_preprocess = std::accumulate(preprocess_times.begin(), preprocess_times.end(), 0.0f) / preprocess_times.size();
            float avg_inference = std::accumulate(inference_times.begin(), inference_times.end(), 0.0f) / inference_times.size();
            float avg_postprocess = std::accumulate(postprocess_times.begin(), postprocess_times.end(), 0.0f) / postprocess_times.size();
            float avg_visualize = std::accumulate(visualize_times.begin(), visualize_times.end(), 0.0f) / visualize_times.size();
            float avg_total = std::accumulate(total_times.begin(), total_times.end(), 0.0f) / total_times.size();
            
            auto minmax_inference = std::minmax_element(inference_times.begin(), inference_times.end());
            auto minmax_total = std::minmax_element(total_times.begin(), total_times.end());
            
            fprintf(stdout, "Performance Statistics (Scene: %s, %zu frames):\n", scene_name.c_str(), inference_times.size());
            fprintf(stdout, "  Load:       avg=%.2f ms\n", avg_load);
            fprintf(stdout, "  Preprocess: avg=%.2f ms\n", avg_preprocess);
            fprintf(stdout, "  Inference:  avg=%.2f ms (min=%.2f, max=%.2f)\n",
                    avg_inference, *minmax_inference.first, *minmax_inference.second);
            fprintf(stdout, "  Postprocess: avg=%.2f ms\n", avg_postprocess);
            fprintf(stdout, "  Visualize:  avg=%.2f ms\n", avg_visualize);
            fprintf(stdout, "  Total:      avg=%.2f ms (min=%.2f, max=%.2f), FPS=%.2f\n",
                    avg_total, *minmax_total.first, *minmax_total.second, 1000.0f / avg_total);
        }
        
        // Create video
        std::string video_path = bevformer::join_path(scene_dir, scene_name + "_result.mp4");
        std::string actual_video_path;
        if (bevformer::create_video_from_images(images_dir, video_path, actual_video_path, args.fps)) {
            std::cout << "Scene " << scene_name << " completed, video: " << actual_video_path << std::endl;
        } else {
            std::cerr << "Warning: Failed to create video for scene " << scene_name << std::endl;
        }
    }
    
    // Cleanup
    middleware::free_io(&io_data);
    AX_ENGINE_DestroyHandle(handle);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    
    return 0;
}

