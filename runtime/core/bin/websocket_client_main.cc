

#include <boost/asio.hpp>
#include <boost/asio/signal_set.hpp>
#include <boost/beast.hpp>
#include <boost/nowide/convert.hpp>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "frontend/wav.h"
#include "onnxruntime_cxx_api.h"
#include "utils/common_sdl.h"
#include "utils/flags.h"
#include "utils/timer.h"
#include "utils/wav_writer.h"
#include "websocket/websocket_client.h"

namespace beast = boost::beast;
namespace websocket = beast::websocket;
using tcp = boost::asio::ip::tcp;

void handle_stop(boost::asio::io_context& io_context,
                 std::thread& server_thread) {
  io_context.stop();
  if (server_thread.joinable()) {
    server_thread.join();
  }
}

class BroadcastSession : public std::enable_shared_from_this<BroadcastSession> {
 public:
  using CloseCallback = std::function<void(std::shared_ptr<BroadcastSession>)>;

  BroadcastSession(tcp::socket socket, CloseCallback callback)
      : ws_(std::move(socket)), close_callback_(std::move(callback)) {}

  void Start() {
    ws_.async_accept([self = shared_from_this()](beast::error_code ec) {
      if (!ec) {
        self->DoRead();
      }
    });
  }

  void Deliver(const std::string& msg) {
    bool write_in_progress = !write_msgs_.empty();
    write_msgs_.push_back(msg);
    if (!write_in_progress) {
      DoWrite();
    }
  }

 private:
  websocket::stream<beast::tcp_stream> ws_;
  std::deque<std::string> write_msgs_;
  beast::flat_buffer buffer_;
  CloseCallback close_callback_;

  void DoRead() {
    ws_.async_read(buffer_, [self = shared_from_this()](beast::error_code ec,
                                                        std::size_t length) {
      if (!ec) {
        self->DoRead();
      } else {
        std::cerr << "Read Error: " << ec.message() << std::endl;
        self->CloseSession();
      }
    });
  }

  void DoWrite() {
    ws_.async_write(boost::asio::buffer(write_msgs_.front()),
                    [self = shared_from_this()](beast::error_code ec,
                                                std::size_t /*length*/) {
                      if (!ec) {
                        self->write_msgs_.pop_front();
                        if (!self->write_msgs_.empty()) {
                          self->DoWrite();
                        }
                      } else {
                        std::cerr << "Write Error: " << ec.message()
                                  << std::endl;
                        self->CloseSession();
                      }
                    });
  }

  void CloseSession() {
    if (ws_.is_open()) {
      beast::error_code ec;
      ws_.close(websocket::close_code::normal, ec);
      if (ec) {
        std::cerr << "Error closing WebSocket: " << ec.message() << std::endl;
      }
    }
    if (close_callback_) {
      close_callback_(shared_from_this());
    }
  }
};

class BroadcastServer {
 public:
  BroadcastServer(boost::asio::io_context& io_context, int port)
      : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
    DoAccept();
  }

  void Broadcast(const std::string& message) {
    for (auto& session : sessions_) {
      session->Deliver(message);
    }
  }

  void RemoveSession(std::shared_ptr<BroadcastSession> session) {
    sessions_.erase(session);
  }

 private:
  void DoAccept() {
    acceptor_.async_accept(
        [this](boost::system::error_code ec, tcp::socket socket) {
          if (!ec) {
            auto session = std::make_shared<BroadcastSession>(
                std::move(socket),
                [this](std::shared_ptr<BroadcastSession> session) {
                  this->RemoveSession(session);
                });
            sessions_.insert(session);
            session->Start();
          } else {
            std::cerr << "Accept Error: " << ec.message() << std::endl;
          }
          DoAccept();
        });
  }

  tcp::acceptor acceptor_;
  std::set<std::shared_ptr<BroadcastSession>> sessions_;
};

class VadIterator {
 public:
  Ort::Env env;
  Ort::SessionOptions session_options;
  std::shared_ptr<Ort::Session> session = nullptr;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

  void init_engine_threads(int inter_threads, int intra_threads) {
    session_options.SetIntraOpNumThreads(intra_threads);
    session_options.SetInterOpNumThreads(inter_threads);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
  }
  void init_onnx_model(const std::wstring& model_path) {
    std::string narrow_model_path = boost::nowide::narrow(model_path);
    init_engine_threads(1, 1);
    session.reset(
        new Ort::Session(env, narrow_model_path.c_str(), session_options));
  }

  void reset_states() {
    std::memset(_h.data(), 0.0f, _h.size() * sizeof(float));
    std::memset(_c.data(), 0.0f, _c.size() * sizeof(float));
    triggered = false;
  }

  bool predict(const std::vector<float>& data) {
    input.assign(data.begin(), data.end());
    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), input_node_dims, 2);
    Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
        memory_info, sr.data(), sr.size(), sr_node_dims, 1);
    Ort::Value h_ort = Ort::Value::CreateTensor<float>(
        memory_info, _h.data(), _h.size(), hc_node_dims, 3);
    Ort::Value c_ort = Ort::Value::CreateTensor<float>(
        memory_info, _c.data(), _c.size(), hc_node_dims, 3);

    ort_inputs.clear();
    ort_inputs.emplace_back(std::move(input_ort));
    ort_inputs.emplace_back(std::move(sr_ort));
    ort_inputs.emplace_back(std::move(h_ort));
    ort_inputs.emplace_back(std::move(c_ort));

    ort_outputs = session->Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
        ort_inputs.size(), output_node_names.data(), output_node_names.size());

    float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
    std::memcpy(_h.data(), ort_outputs[1].GetTensorMutableData<float>(),
                size_hc * sizeof(float));
    std::memcpy(_c.data(), ort_outputs[2].GetTensorMutableData<float>(),
                size_hc * sizeof(float));

    return speech_prob >= threshold;
  }

 public:
  bool process(const std::vector<float>& input_wav) {
    // reset_states();
    // int audio_length_samples = input_wav.size();
    // for (int j = 0; j < audio_length_samples; j += window_size_samples) {
    //   if (j + window_size_samples > audio_length_samples)
    //     break;
    //   std::vector<float> r{&input_wav[0] + j,
    //                        &input_wav[0] + j + window_size_samples};
    bool speech = predict(input_wav);
    if (speech) {
      triggered = true;
      return true;
    } else {
      if (triggered) {
        reset_states();
      }
      return false;
    }
  }

 private:
  int64_t window_size_samples;
  int sample_rate;
  float threshold;
  unsigned int size_hc = 2 * 1 * 64;
  std::vector<Ort::Value> ort_inputs;
  std::vector<float> input, _h, _c;
  std::vector<int64_t> sr;
  int64_t input_node_dims[2] = {};
  const int64_t sr_node_dims[1] = {1};
  const int64_t hc_node_dims[3] = {2, 1, 64};
  std::vector<Ort::Value> ort_outputs;
  std::vector<const char*> input_node_names = {"input", "sr", "h", "c"};
  std::vector<const char*> output_node_names = {"output", "hn", "cn"};

 public:
  bool triggered;
  VadIterator(const std::wstring ModelPath, int Sample_rate = 16000,
              int windows_frame_size = 64, float Threshold = 0.5)
      : triggered(false), threshold(Threshold), sample_rate(Sample_rate) {
    init_onnx_model(ModelPath);
    window_size_samples = windows_frame_size * (sample_rate / 1000);
    input.resize(window_size_samples);
    input_node_dims[0] = 1;
    input_node_dims[1] = window_size_samples;
    _h.resize(size_hc);
    _c.resize(size_hc);
    sr.resize(1);
    sr[0] = sample_rate;
  }
};

DEFINE_string(hostname, "127.0.0.1", "hostname of websocket server");
DEFINE_int32(port, 10086, "port of asr server");
DEFINE_int32(port_rebroadcast, 10087, "port of local rebroadcast server");
DEFINE_int32(nbest, 1, "n-best of decode result");
DEFINE_string(wav_path, "", "test wav file path");
DEFINE_bool(continuous_decoding, false, "continuous decoding mode");
DEFINE_int32(device, -1, "audio device id");
DEFINE_bool(save_audio, false, "save audio to wav file");
DEFINE_double(vad_threshold, 0.5, "vad threshold");

int main(int argc, char* argv[]) {
  boost::asio::io_context io_context;
  // Initialize broadcast_server with the shared pointer
  auto broadcast_server =
      std::make_shared<BroadcastServer>(io_context, FLAGS_port_rebroadcast);

  // Capture io_context and broadcast_server by reference in the lambda
  std::thread server_thread(
      [&io_context, &broadcast_server]() { io_context.run(); });

  boost::asio::signal_set signals(io_context, SIGINT, SIGTERM);
  signals.async_wait(
      [&](const boost::system::error_code& /*error*/, int /*signal_number*/) {
        handle_stop(io_context, server_thread);
      });

  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  wenet::WebSocketClient client(FLAGS_hostname, FLAGS_port);
  client.set_nbest(FLAGS_nbest);
  client.set_continuous_decoding(FLAGS_continuous_decoding);
  client.SendStartSignal();
  client.SetMessageCallback([broadcast_server](const std::string& message) {
    broadcast_server->Broadcast(message);
  });

  const int sample_rate = 16000;
  std::wstring path = L"silero_vad.onnx";
  VadIterator vad(path, sample_rate, 64, FLAGS_vad_threshold);

  if (FLAGS_wav_path.empty()) {
    wav_writer wavWriter;

    // Save WAV file if required
    if (FLAGS_save_audio) {
      time_t now = time(0);
      char buffer[80];
      strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
      std::string filename = std::string(buffer) + ".wav";

      wavWriter.open(filename, sample_rate, 16, 1);
    }

    audio_async audio(300);  // Buffer duration in milliseconds
    if (!audio.init(FLAGS_device, sample_rate)) {
      std::cerr << "Failed to initialize audio!" << std::endl;
      SDL_Quit();
      return 1;
    }

    audio.resume();
    bool is_running = true;
    bool is_speech = false;
    std::vector<int16_t> audio_data;
    std::vector<int16_t> excess_data;
    int desired_frame_count = 1024;
    const int N = 4;
    std::vector<int16_t> pre_speech_buffer;
    auto desired_interval_ms = std::chrono::milliseconds(64);
    std::chrono::high_resolution_clock::time_point last_data_time =
        std::chrono::high_resolution_clock::now();

    while (is_running) {
      is_running = sdl_poll_events();
      audio.get(audio_data);

      if (audio_data.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      last_data_time = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < audio_data.size(); i += desired_frame_count) {
        if (i + desired_frame_count > audio_data.size()) {
          break;
        }
        std::vector<float> r;
        r.reserve(desired_frame_count);  // Reserve space to avoid reallocations
        for (int j = i; j < i + desired_frame_count; ++j) {
          // Normalize int16_t to float range [-1.0, 1.0]
          r.push_back(audio_data[j] / 32768.0f);
        }

        if (vad.process(r)) {
          is_speech = true;
          // VLOG(2) << "Speech detected";
          if (!pre_speech_buffer.empty()) {
            VLOG(2) << "Sending pre-speech buffer: " << pre_speech_buffer.size();
            client.SendBinaryData(pre_speech_buffer.data(),
                                  pre_speech_buffer.size() * sizeof(int16_t));
            if (FLAGS_save_audio) {
              wavWriter.write(pre_speech_buffer.data(),
                              pre_speech_buffer.size());
            }
            pre_speech_buffer.clear();
          }

          VLOG(2) << "Sending speech data: " << i;
          client.SendBinaryData(audio_data.data() + i,
                                desired_frame_count * sizeof(int16_t));

          if (FLAGS_save_audio) {
            wavWriter.write(audio_data.data() + i, desired_frame_count);
          }

        } else {
          if (is_speech) {
            // Detected end of speech
            std::vector<int16_t> end_of_speech_buffer(
                audio_data.begin() + i,
                audio_data.begin() + i + desired_frame_count);

            // Append 1024 empty samples to the end of speech data
            std::vector<int16_t> empty_samples(1024,
                                               0);  // 1024 samples of silence
            end_of_speech_buffer.insert(end_of_speech_buffer.end(),
                                        empty_samples.begin(),
                                        empty_samples.end());

            // Send the end of speech data with the empty samples appended
            client.SendBinaryData(
                end_of_speech_buffer.data(),
                end_of_speech_buffer.size() * sizeof(int16_t));

            if (FLAGS_save_audio) {
              wavWriter.write(end_of_speech_buffer.data(),
                              end_of_speech_buffer.size());
            }

            client.SendEndSignal();
            is_speech = false;
          }
          // VLOG(2) << "Silence detected";
          // VLOG(2) << "Pre-speech buffer size: " << pre_speech_buffer.size();
          std::vector<int16_t> temp;
          int buffer_size = N * desired_frame_count;
          int start_index =
              std::max(0, (int)pre_speech_buffer.size() -
                              (buffer_size - desired_frame_count));
          temp.insert(temp.end(), pre_speech_buffer.begin() + start_index,
                      pre_speech_buffer.end());
          temp.insert(temp.end(), audio_data.begin() + i,
                      audio_data.begin() + i + desired_frame_count);
          pre_speech_buffer = std::move(temp);
        }
      }

      // if(vad.process(std::vector<float>(audio_data.begin(),
      // audio_data.end()))) {
      //   client.SendBinaryData(audio_data.data(), audio_data.size() *
      //   sizeof(int16_t));
      // } else {
      //   VLOG(2) << "Silence detected";
      // }

      // if (FLAGS_save_audio) {
      //   wavWriter.write(audio_data.data(), audio_data.size());
      // }

      auto now = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> time_diff =
          now - last_data_time;

      auto time_to_wait = desired_interval_ms - time_diff;
      // VLOG(2) << "sleeping " << time_to_wait.count() << "ms";
      if (time_to_wait > std::chrono::milliseconds(0)) {
        std::this_thread::sleep_for(time_to_wait);
      }
    }
  } else {
    wenet::WavReader wav_reader(FLAGS_wav_path);
    // Only support 16K
    CHECK_EQ(wav_reader.sample_rate(), sample_rate);
    const int num_samples = wav_reader.num_samples();
    // Send data every 0.5 second
    const float interval = 0.064;
    const int sample_interval = interval * sample_rate;
    for (int start = 0; start < num_samples; start += sample_interval) {
      if (client.done()) {
        break;
      }
      int end = std::min(start + sample_interval, num_samples);
      // Convert to short
      std::vector<int16_t> data;
      data.reserve(end - start);
      for (int j = start; j < end; j++) {
        data.push_back(static_cast<int16_t>(wav_reader.data()[j]));
      }
      // TODO(Binbin Zhang): Network order?
      // Send PCM data
      client.SendBinaryData(data.data(), data.size() * sizeof(int16_t));
      VLOG(2) << "Send " << data.size() << " samples";
      std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<int>(64)));
    }
    wenet::Timer timer;
    client.SendEndSignal();
    client.Join();
    VLOG(2) << "Total latency: " << timer.Elapsed() << "ms.";
  }
  handle_stop(io_context, server_thread);
  return 0;
}
