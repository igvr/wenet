#include <iostream>
#include <vector>
#include <string>
#include "onnxruntime_cxx_api.h"
#include "utils/wav_reader.h"
#include "utils/flags.h"

// Define VAD Iterator Class
class VadIterator {
 public:
  Ort::Env env;
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session = nullptr;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
  std::vector<float> _h, _c; // Hidden and cell state for LSTM
  std::vector<const char*> input_node_names = {"input", "sr", "h", "c"};
  std::vector<const char*> output_node_names = {"output", "hn", "cn"};
  std::vector<int64_t> input_node_dims = {1, 0}; // {batch, sequence length}
  const int64_t sr_node_dims[1] = {1};
  const int64_t hc_node_dims[3] = {2, 1, 64}; // LSTM dimension
  float threshold;
  size_t size_hc = 2 * 1 * 64; // LSTM state size

  VadIterator(const std::string& model_path, float Threshold = 0.5)
      : threshold(Threshold) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
    _h.resize(size_hc, 0);
    _c.resize(size_hc, 0);
    input_node_dims[1] = 1024; // Set sequence length to 1024 frames
  }

  bool predict(const std::vector<float>& data) {
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(data.data()), data.size(), input_node_dims.data(), 2);
    std::vector<int64_t> sr = {16000}; // Sample rate
    Ort::Value sr_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, sr.data(), sr.size(), sr_node_dims, 1);
    Ort::Value h_tensor = Ort::Value::CreateTensor<float>(memory_info, _h.data(), _h.size(), hc_node_dims, 3);
    Ort::Value c_tensor = Ort::Value::CreateTensor<float>(memory_info, _c.data(), _c.size(), hc_node_dims, 3);
    std::array<Ort::Value, 4> input_tensors = {input_tensor, sr_tensor, h_tensor, c_tensor};

    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(), output_node_names.data(), output_node_names.size());
    float speech_prob = output_tensors[0].GetTensorMutableData<float>()[0];
    std::memcpy(_h.data(), output_tensors[1].GetTensorMutableData<float>(), size_hc * sizeof(float));
    std::memcpy(_c.data(), output_tensors[2].GetTensorMutableData<float>(), size_hc * sizeof(float));
    return speech_prob >= threshold;
  }
};

// Main Function
int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model_path> <wav_path>" << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  std::string wav_path = argv[2];

  VadIterator vad(model_path);
  wenet::WavReader wav_reader(wav_path);
  const auto& data = wav_reader.Data();
  int sample_rate = wav_reader.SampleRate();
  if (sample_rate != 16000) {
    std::cerr << "Sample rate of the WAV file must be 16000 Hz." << std::endl;
    return 1;
  }

  const int frame_size = 1024; // Number of frames to process at once
  std::vector<float> frame_data(frame_size, 0.0f);

  std::cout << "VAD Results: ";
  for (size_t i = 0; i < data.size(); i += frame_size) {
    size_t end = std::min(data.size(), i + frame_size);
    std::transform(data.begin() + i, data.begin() + end, frame_data.begin(),
                   [](int16_t val) { return val / 32768.0f; }); // Normalize int16_t to float

    bool is_speech = vad.predict(frame_data);
    std::cout << (is_speech ? "+" : "-");
  }
  std::cout << std::endl;

  return 0;
}
