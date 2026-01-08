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

#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <cstdio>
#include <algorithm>

// Simple JSON parser (simplified - for production use jsoncpp or similar)
// This is a basic implementation - you may want to use a proper JSON library
namespace simple_json {
    bool parse_value(const std::string& json_str, size_t& pos, std::string& key, std::string& value);
    bool parse_array(const std::string& json_str, size_t& pos, std::vector<float>& arr);
    bool parse_number(const std::string& json_str, size_t& pos, float& num);
}

namespace bevformer {

// Helper function to extract array from JSON string
static std::vector<float> extract_array_from_json(const std::string& json_str, const std::string& key) {
    std::vector<float> result;
    size_t key_pos = json_str.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    
    size_t arr_start = json_str.find("[", key_pos);
    size_t arr_end = json_str.find("]", arr_start);
    if (arr_start == std::string::npos || arr_end == std::string::npos) return result;
    
    std::string arr_str = json_str.substr(arr_start + 1, arr_end - arr_start - 1);
    std::istringstream iss(arr_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\n"));
        token.erase(token.find_last_not_of(" \t\n") + 1);
        if (token.empty()) continue;
        try {
            result.push_back(std::stof(token));
        } catch (...) {
            // Skip invalid numbers
        }
    }
    return result;
}

// Helper function to extract number from JSON string
static float extract_float_from_json(const std::string& json_str, const std::string& key, float default_val = 0.0f) {
    size_t key_pos = json_str.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return default_val;
    
    size_t colon_pos = json_str.find(":", key_pos);
    if (colon_pos == std::string::npos) return default_val;
    
    size_t val_start = json_str.find_first_not_of(" \t\n", colon_pos + 1);
    if (val_start == std::string::npos) return default_val;
    
    size_t val_end = val_start;
    while (val_end < json_str.length() && 
           (std::isdigit(json_str[val_end]) || json_str[val_end] == '.' || 
            json_str[val_end] == '-' || json_str[val_end] == '+' || json_str[val_end] == 'e' || json_str[val_end] == 'E')) {
        val_end++;
    }
    
    if (val_end > val_start) {
        try {
            return std::stof(json_str.substr(val_start, val_end - val_start));
        } catch (...) {
            return default_val;
        }
    }
    return default_val;
}

// Helper function to extract int from JSON string (unused but kept for potential future use)
// static int extract_int_from_json(const std::string& json_str, const std::string& key, int default_val = 0) {
//     return static_cast<int>(extract_float_from_json(json_str, key, static_cast<float>(default_val)));
// }

// Helper function to extract nested value (e.g., "model.transformer.bev_h")
static float extract_nested_float(const std::string& json_str, const std::vector<std::string>& path, float default_val = 0.0f) {
    size_t pos = 0;
    for (size_t i = 0; i < path.size(); ++i) {
        size_t key_pos = json_str.find("\"" + path[i] + "\"", pos);
        if (key_pos == std::string::npos) return default_val;
        
        if (i < path.size() - 1) {
            // Not the last key, find the opening brace
            size_t brace_pos = json_str.find("{", key_pos);
            if (brace_pos == std::string::npos) return default_val;
            pos = brace_pos + 1;
        } else {
            // Last key, extract the value
            return extract_float_from_json(json_str.substr(key_pos), path[i], default_val);
        }
    }
    return default_val;
}

static std::vector<float> extract_nested_array(const std::string& json_str, const std::vector<std::string>& path) {
    size_t pos = 0;
    for (size_t i = 0; i < path.size(); ++i) {
        size_t key_pos = json_str.find("\"" + path[i] + "\"", pos);
        if (key_pos == std::string::npos) return std::vector<float>();
        
        if (i < path.size() - 1) {
            // Not the last key, find the opening brace
            size_t brace_pos = json_str.find("{", key_pos);
            if (brace_pos == std::string::npos) return std::vector<float>();
            pos = brace_pos + 1;
        } else {
            // Last key, extract the array
            return extract_array_from_json(json_str.substr(key_pos), path[i]);
        }
    }
    return std::vector<float>();
}

bool load_config_from_json(const std::string& config_path, Config& config) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        return false;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    
    // Parse model.transformer parameters
    config.bev_h = static_cast<int>(extract_nested_float(json_content, {"model", "transformer", "bev_h"}, 200.0f));
    config.bev_w = static_cast<int>(extract_nested_float(json_content, {"model", "transformer", "bev_w"}, 200.0f));
    config.embed_dims = static_cast<int>(extract_nested_float(json_content, {"model", "transformer", "embed_dims"}, 256.0f));
    
    // Parse model.bbox_coder parameters
    std::vector<float> pc_range = extract_nested_array(json_content, {"model", "bbox_coder", "pc_range"});
    if (!pc_range.empty() && pc_range.size() >= 6) {
        config.pc_range = pc_range;
    }
    
    std::vector<float> post_center_range = extract_nested_array(json_content, {"model", "bbox_coder", "post_center_range"});
    if (!post_center_range.empty() && post_center_range.size() >= 6) {
        config.post_center_range = post_center_range;
    }
    
    config.max_num = static_cast<int>(extract_nested_float(json_content, {"model", "bbox_coder", "max_num"}, 100.0f));
    config.score_threshold = extract_nested_float(json_content, {"model", "bbox_coder", "score_threshold"}, 0.1f);
    
    // Parse img_norm parameters
    std::vector<float> img_mean = extract_nested_array(json_content, {"img_norm", "mean"});
    if (!img_mean.empty() && img_mean.size() >= 3) {
        config.img_mean = img_mean;
    }
    
    std::vector<float> img_std = extract_nested_array(json_content, {"img_norm", "std"});
    if (!img_std.empty() && img_std.size() >= 3) {
        config.img_std = img_std;
    }
    
    // Parse to_rgb (boolean)
    size_t to_rgb_pos = json_content.find("\"to_rgb\"");
    if (to_rgb_pos != std::string::npos) {
        size_t val_start = json_content.find_first_not_of(" \t\n:", to_rgb_pos + 8);
        if (val_start != std::string::npos) {
            std::string val_str = json_content.substr(val_start, 5);  // "true" or "false"
            config.to_rgb = (val_str.find("true") == 0);
        }
    }
    
    // Print loaded config (only once)
    static bool config_loaded_shown = false;
    if (!config_loaded_shown) {
        std::cout << "Loaded configuration:" << std::endl;
        std::cout << "  bev_h=" << config.bev_h << ", bev_w=" << config.bev_w 
                  << ", embed_dims=" << config.embed_dims << std::endl;
        std::cout << "  max_num=" << config.max_num 
                  << ", score_threshold=" << config.score_threshold << std::endl;
        std::cout << "  pc_range=[" << config.pc_range[0] << ", " << config.pc_range[1] 
                  << ", " << config.pc_range[2] << ", " << config.pc_range[3] 
                  << ", " << config.pc_range[4] << ", " << config.pc_range[5] << "]" << std::endl;
        std::cout << "  post_center_range=[" << config.post_center_range[0] << ", " 
                  << config.post_center_range[1] << ", " << config.post_center_range[2] 
                  << ", " << config.post_center_range[3] << ", " << config.post_center_range[4] 
                  << ", " << config.post_center_range[5] << "]" << std::endl;
        config_loaded_shown = true;
    }
    
    return true;
}

bool load_scene_index(const std::string& data_dir,
                     std::map<std::string, std::vector<int>>& scenes) {
    std::string scene_index_path = join_path(data_dir, "scene_index.json");
    std::ifstream file(scene_index_path);
    
    // Try to parse JSON file first
    if (file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        file.close();
        
        // Simple JSON parsing for scene_index.json
        // Look for "scenes": { "scene_token": { "samples": [...] } }
        size_t scenes_pos = content.find("\"scenes\"");
        if (scenes_pos != std::string::npos) {
            // Find the opening brace after "scenes"
            size_t brace_start = content.find("{", scenes_pos);
            if (brace_start != std::string::npos) {
                size_t pos = brace_start + 1;
                
                // Extract scene names and samples
                // Pattern: "scene_name": { "samples": [...] }
                while (pos < content.length()) {
                    // Find next quoted string (scene name)
                    size_t quote_pos = content.find("\"", pos);
                    if (quote_pos == std::string::npos) break;
                    
                    // Check if this is a key (followed by ":")
                    size_t colon_pos = content.find(":", quote_pos);
                    if (colon_pos == std::string::npos || colon_pos > quote_pos + 200) {
                        pos = quote_pos + 1;
                        continue;
                    }
                    
                    size_t name_start = quote_pos + 1;
                    size_t name_end = content.find("\"", name_start);
                    if (name_end == std::string::npos || name_end >= colon_pos) {
                        pos = quote_pos + 1;
                        continue;
                    }
                    
                    std::string scene_name = content.substr(name_start, name_end - name_start);
                    
                    // Find "samples" key after this scene name (within reasonable distance)
                    size_t search_end = std::min(name_end + 1000, content.length());
                    size_t samples_pos = content.find("\"samples\"", name_end, search_end - name_end);
                    if (samples_pos != std::string::npos) {
                        size_t arr_start = content.find("[", samples_pos);
                        size_t arr_end = content.find("]", arr_start);
                        if (arr_start != std::string::npos && arr_end != std::string::npos) {
                            std::string arr_str = content.substr(arr_start + 1, arr_end - arr_start - 1);
                            std::istringstream iss(arr_str);
                            std::string token;
                            std::vector<int> samples;
                            while (std::getline(iss, token, ',')) {
                                token.erase(0, token.find_first_not_of(" \t\n"));
                                token.erase(token.find_last_not_of(" \t\n") + 1);
                                if (token.empty()) continue;
                                try {
                                    samples.push_back(std::stoi(token));
                                } catch (...) {
                                    // Skip invalid numbers
                                }
                            }
                            if (!samples.empty()) {
                                scenes[scene_name] = samples;
                                std::cout << "Parsed scene from JSON: " << scene_name << " with " << samples.size() << " frames" << std::endl;
                            }
                        }
                        // Move past this scene's closing brace
                        size_t next_brace = content.find("}", arr_end);
                        if (next_brace != std::string::npos) {
                            pos = next_brace + 1;
                        } else {
                            pos = arr_end + 1;
                        }
                    } else {
                        pos = name_end + 1;
                    }
                }
            }
            
            if (!scenes.empty()) {
                std::cout << "Loaded " << scenes.size() << " scenes from scene_index.json" << std::endl;
                return true;
            } else {
                // Warning only shown once
                static bool scenes_warning_shown = false;
                if (!scenes_warning_shown) {
                    std::cerr << "Warning: Found 'scenes' in JSON but could not parse scene data" << std::endl;
                    scenes_warning_shown = true;
                }
            }
        } else {
            std::cerr << "Warning: 'scenes' key not found in scene_index.json" << std::endl;
        }
    }
    
    // Fallback: scan directory structure
    static bool scan_warning_shown = false;
    if (!scan_warning_shown) {
        std::cerr << "Warning: Using directory scan for scene index. Consider using jsoncpp." << std::endl;
        scan_warning_shown = true;
    }
    
    // Scan data_dir for scene folders (any directory that contains meta_*.json files)
    // First, find all directories (excluding the data_dir itself)
    std::string cmd = "find \"" + data_dir + "\" -maxdepth 1 -type d ! -path \"" + data_dir + "\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to scan directory: " << data_dir << std::endl;
        return false;
    }
    
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        std::string scene_path(buffer);
        // Remove newline
        scene_path.erase(scene_path.find_last_not_of(" \n\r\t") + 1);
        if (scene_path.empty()) continue;
        
        // Extract scene name
        size_t last_slash = scene_path.find_last_of("/");
        if (last_slash == std::string::npos) continue;
        
        std::string scene_name = scene_path.substr(last_slash + 1);
        
        // Skip scene_index.json and other files
        if (scene_name.find(".") != std::string::npos) continue;
        
        // Find all meta files in this scene directory
        std::string find_cmd = "find " + scene_path + " -maxdepth 1 -name 'meta_*.json' 2>/dev/null | sort";
        FILE* meta_pipe = popen(find_cmd.c_str(), "r");
        if (meta_pipe) {
            std::vector<int> samples;
            char meta_buffer[512];
            while (fgets(meta_buffer, sizeof(meta_buffer), meta_pipe)) {
                std::string meta_path(meta_buffer);
                meta_path.erase(meta_path.find_last_not_of(" \n\r\t") + 1);
                if (meta_path.empty()) continue;
                
                // Extract frame index from filename: meta_XXXXXX.json
                size_t meta_start = meta_path.find("meta_");
                size_t meta_end = meta_path.find(".json");
                if (meta_start != std::string::npos && meta_end != std::string::npos) {
                    std::string idx_str = meta_path.substr(meta_start + 5, meta_end - meta_start - 5);
                    try {
                        int frame_idx = std::stoi(idx_str);
                        samples.push_back(frame_idx);
                    } catch (...) {
                        // Skip invalid numbers
                    }
                }
            }
            pclose(meta_pipe);
            
            if (!samples.empty()) {
                std::sort(samples.begin(), samples.end());
                scenes[scene_name] = samples;
                std::cout << "Found scene: " << scene_name << " with " << samples.size() << " frames" << std::endl;
            }
        }
    }
    pclose(pipe);
    
    if (scenes.empty()) {
        std::cerr << "No scenes found in data directory: " << data_dir << std::endl;
        return false;
    }
    
    std::cout << "Found " << scenes.size() << " scenes by directory scan" << std::endl;
    return true;
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

std::string join_path(const std::string& dir, const std::string& filename) {
    if (dir.empty()) return filename;
    if (dir.back() == '/') return dir + filename;
    return dir + "/" + filename;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> sigmoid_vec(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid(x[i]);
    }
    return result;
}

void print_progress_bar(size_t current, size_t total, const std::string& prefix, 
                        float elapsed_ms) {
    if (total == 0) return;
    
    const int bar_width = 40;
    float progress = static_cast<float>(current) / static_cast<float>(total);
    int pos = static_cast<int>(bar_width * progress);
    
    std::string bar;
    bar.reserve(bar_width + 10);
    bar += "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) {
            bar += "=";
        } else if (i == pos) {
            bar += ">";
        } else {
            bar += " ";
        }
    }
    bar += "]";
    
    int percent = static_cast<int>(progress * 100.0f);
    
    // Calculate speed (frames per second)
    float fps = 0.0f;
    float eta_seconds = 0.0f;
    if (elapsed_ms > 0 && current > 0) {
        float elapsed_seconds = elapsed_ms / 1000.0f;
        fps = static_cast<float>(current) / elapsed_seconds;
        if (current < total) {
            float remaining = static_cast<float>(total - current);
            eta_seconds = remaining / fps;
        }
    }
    
    if (current < total && fps > 0) {
        // Show progress with ETA
        int eta_min = static_cast<int>(eta_seconds / 60);
        int eta_sec = static_cast<int>(eta_seconds) % 60;
        fprintf(stdout, "\r%s %s %3d%% [%zu/%zu] %.1ffps, ETA: %02d:%02d", 
                prefix.c_str(), bar.c_str(), percent, current, total, fps, eta_min, eta_sec);
    } else {
        // Show final progress
        fprintf(stdout, "\r%s %s %3d%% [%zu/%zu]", 
                prefix.c_str(), bar.c_str(), percent, current, total);
    }
    fflush(stdout);
    
    if (current == total) {
        fprintf(stdout, "\n");
    }
}

} // namespace bevformer

