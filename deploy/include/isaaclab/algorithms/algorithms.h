// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

inline void require_exact_ort_input_size(
    const std::string& input_name,
    size_t actual_size,
    size_t expected_size)
{
    if (actual_size != expected_size) {
        throw std::runtime_error(
            "ONNX input '" + input_name + "' size mismatch: expected "
            + std::to_string(expected_size) + ", got "
            + std::to_string(actual_size));
    }
}

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
    explicit OrtRunner(std::string model_path, size_t expected_action_size = 0)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        if (session->GetInputCount() == 0 || session->GetOutputCount() != 1) {
            throw std::runtime_error(
                "Policy ONNX must have at least one input and exactly one output");
        }
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            const auto tensor_info = input_type.GetTensorTypeAndShapeInfo();
            if (tensor_info.GetElementType()
                != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                throw std::runtime_error("Policy ONNX inputs must be float32");
            }
            input_shapes.push_back(tensor_info.GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (auto& shape : input_shapes) {
            size_t size = 1;
            for (size_t axis = 0; axis < shape.size(); ++axis) {
                auto& dim = shape[axis];
                // The controller always runs a single robot. Resolve only a
                // dynamic batch axis; every feature axis must be concrete.
                if (axis == 0 && dim == -1) {
                    dim = 1;
                }
                if (dim <= 0) {
                    throw std::runtime_error(
                        "ONNX input has unresolved dimension at axis "
                        + std::to_string(axis));
                }
                const size_t extent = static_cast<size_t>(dim);
                if (size > std::numeric_limits<size_t>::max() / extent) {
                    throw std::runtime_error("ONNX input tensor size overflow");
                }
                size *= extent;
            }
            input_sizes.push_back(size);
        }

        // Get output shape
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        const auto output_info = output_type.GetTensorTypeAndShapeInfo();
        if (output_info.GetElementType()
            != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Policy ONNX output must be float32");
        }
        output_shape = output_info.GetShape();
        if (output_shape.size() != 2) {
            throw std::runtime_error(
                "Policy ONNX output must have rank 2 [batch, action]");
        }
        if (output_shape[0] == -1) {
            output_shape[0] = 1;
        }
        if (output_shape[0] != 1 || output_shape[1] <= 0) {
            throw std::runtime_error(
                "Policy ONNX output must resolve to [1, positive_action_dim]");
        }
        action_size_ = static_cast<size_t>(output_shape[1]);
        if (expected_action_size != 0 && action_size_ != expected_action_size) {
            throw std::runtime_error(
                "Policy ONNX action dimension mismatch: expected "
                + std::to_string(expected_action_size) + ", got "
                + std::to_string(action_size_));
        }
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());

        action.resize(action_size_);
    }

    size_t output_size() const noexcept { return action_size_; }
    size_t input_count() const noexcept { return input_sizes.size(); }
    size_t input_size(size_t index = 0) const
    {
        if (index >= input_sizes.size()) {
            throw std::out_of_range("Policy ONNX input index is out of range");
        }
        return input_sizes[index];
    }
    std::string input_name(size_t index = 0) const
    {
        if (index >= input_names.size()) {
            throw std::out_of_range("Policy ONNX input-name index is out of range");
        }
        return std::string(input_names[index]);
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
        for(size_t i = 0; i < input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            require_exact_ort_input_size(
                name_str, input_data.size(), input_sizes[i]);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        if (output_tensor.size() != 1 || !output_tensor.front().IsTensor()) {
            throw std::runtime_error("Policy ONNX inference returned an invalid output");
        }
        const auto runtime_info = output_tensor.front().GetTensorTypeAndShapeInfo();
        if (runtime_info.GetElementType()
            != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || runtime_info.GetElementCount() != action_size_) {
            throw std::runtime_error(
                "Policy ONNX runtime output size/type changed from its audited contract");
        }
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        for (size_t index = 0; index < action_size_; ++index) {
            if (!std::isfinite(floatarr[index])) {
                throw std::runtime_error(
                    "Policy ONNX produced a non-finite action at index "
                    + std::to_string(index));
            }
        }
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, action_size_ * sizeof(float));
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
    std::vector<size_t> input_sizes;
    std::vector<int64_t> output_shape;
    size_t action_size_ = 0;
};


/** Small raw-observation -> scalar ONNX runner used by deploy estimators.
 *
 * Unlike the action runner above, this accepts a dynamic batch dimension and
 * requires exactly one scalar output. Exported estimators contain their own
 * feature normalization, so C++ feeds each model's exact causal prefix.
 */
class ScalarOrtRunner
{
public:
    explicit ScalarOrtRunner(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "scalar_estimator")
    {
        options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options_);
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() < 1) {
            throw std::runtime_error(
                "Scalar estimator must have one input and at least one output.");
        }
        input_name_ = session_->GetInputNameAllocated(0, allocator_).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator_).get();
        const auto input_type = session_->GetInputTypeInfo(0);
        const auto input_info = input_type.GetTensorTypeAndShapeInfo();
        if (input_info.GetElementType()
            != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Scalar estimator input must be float32.");
        }
        input_shape_ = input_info.GetShape();
        if (input_shape_.empty()) {
            throw std::runtime_error("Scalar estimator input has no dimensions.");
        }
        input_size_ = 1;
        for (size_t axis = 0; axis < input_shape_.size(); ++axis) {
            auto& dim = input_shape_[axis];
            if (axis == 0 && dim == -1) {
                dim = 1;
            }
            if (dim <= 0) {
                throw std::runtime_error(
                    "Scalar estimator has unresolved input dimension at axis "
                    + std::to_string(axis));
            }
            const size_t extent = static_cast<size_t>(dim);
            if (input_size_ > std::numeric_limits<size_t>::max() / extent) {
                throw std::runtime_error(
                    "Scalar estimator input tensor size overflow.");
            }
            input_size_ *= extent;
        }

        const auto output_type = session_->GetOutputTypeInfo(0);
        const auto output_info = output_type.GetTensorTypeAndShapeInfo();
        if (output_info.GetElementType()
            != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Scalar estimator output must be float32.");
        }
        auto output_shape = output_info.GetShape();
        size_t output_size = 1;
        for (size_t axis = 0; axis < output_shape.size(); ++axis) {
            auto& dim = output_shape[axis];
            if (axis == 0 && dim == -1) {
                dim = 1;
            }
            if (dim <= 0) {
                throw std::runtime_error(
                    "Scalar estimator has unresolved output dimension at axis "
                    + std::to_string(axis));
            }
            const size_t extent = static_cast<size_t>(dim);
            if (output_size > std::numeric_limits<size_t>::max() / extent) {
                throw std::runtime_error(
                    "Scalar estimator output tensor size overflow.");
            }
            output_size *= extent;
        }
        if (output_size != 1) {
            throw std::runtime_error(
                "Scalar estimator output must contain exactly one scalar; got "
                + std::to_string(output_size));
        }
    }

    size_t input_size() const { return input_size_; }

    float infer(const std::vector<float>& observation)
    {
        if (observation.size() != input_size_) {
            throw std::runtime_error(
                "Scalar estimator input mismatch: expected " + std::to_string(input_size_) +
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
