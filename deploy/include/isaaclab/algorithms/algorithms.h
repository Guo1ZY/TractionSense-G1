// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }

    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get output shape
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};


/** Small raw-observation -> scalar ONNX runner used by the friction estimator.
 *
 * Unlike the action runner above, this accepts a dynamic batch dimension and
 * a one-dimensional output.  The exported estimator already contains its
 * feature normalization, so C++ feeds the exact 640-D policy prefix.
 */
class ScalarOrtRunner
{
public:
    explicit ScalarOrtRunner(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "friction_estimator")
    {
        options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options_);
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() < 1) {
            throw std::runtime_error("Friction estimator must have one input and at least one output.");
        }
        input_name_ = session_->GetInputNameAllocated(0, allocator_).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator_).get();
        input_shape_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape_.empty()) {
            throw std::runtime_error("Friction estimator input has no dimensions.");
        }
        input_shape_[0] = 1;  // exported batch axis is dynamic
        input_size_ = 1;
        for (const auto dim : input_shape_) {
            if (dim <= 0) {
                throw std::runtime_error("Friction estimator has unresolved non-batch input dimension.");
            }
            input_size_ *= static_cast<size_t>(dim);
        }
    }

    size_t input_size() const { return input_size_; }

    float infer(const std::vector<float>& observation)
    {
        if (observation.size() != input_size_) {
            throw std::runtime_error(
                "Friction estimator input mismatch: expected " + std::to_string(input_size_) +
                ", got " + std::to_string(observation.size()));
        }
        auto memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input = Ort::Value::CreateTensor<float>(
            memory,
            const_cast<float*>(observation.data()),
            observation.size(),
            input_shape_.data(),
            input_shape_.size());
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        auto output = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
        return output.front().GetTensorData<float>()[0];
    }

private:
    Ort::Env env_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    size_t input_size_ = 0;
};
};
