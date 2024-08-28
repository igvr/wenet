#ifndef WAV_WRITER_H
#define WAV_WRITER_H

#include <cstdint>
#include <fstream>
#include <string>

class wav_writer {
 public:
  wav_writer() = default;
  ~wav_writer();

  // Move constructor
  wav_writer(wav_writer&& other) noexcept;
  // Move assignment operator
  wav_writer& operator=(wav_writer&& other) noexcept;

  bool open(const std::string& filename, const uint32_t sample_rate,
            const uint16_t bits_per_sample, const uint16_t channels);
  bool write(const int16_t* data, size_t length);
  bool close();

  // Delete copy constructor and copy assignment to ensure the class is
  // move-only
  wav_writer(const wav_writer&) = delete;
  wav_writer& operator=(const wav_writer&) = delete;

 private:
  std::ofstream file;
  uint32_t dataSize = 0;
  std::string wav_filename;

  bool write_header(const uint32_t sample_rate, const uint16_t bits_per_sample,
                    const uint16_t channels);
  bool write_audio(const int16_t* data, size_t length);
  bool open_wav(const std::string& filename);
};

#endif  // WAV_WRITER_H