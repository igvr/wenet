#include "wav_writer.h"

wav_writer::~wav_writer() {
  if (file.is_open()) {
    file.close();
  }
}

wav_writer::wav_writer(wav_writer&& other) noexcept
    : file(std::move(other.file)),
      dataSize(other.dataSize),
      wav_filename(std::move(other.wav_filename)) {
  // Reset the state of 'other' to a valid but empty state
  other.dataSize = 0;
  other.wav_filename = "";
}

wav_writer& wav_writer::operator=(wav_writer&& other) noexcept {
  if (this != &other) {
    if (file.is_open()) {
      file.close();
    }
    file = std::move(other.file);
    dataSize = other.dataSize;
    wav_filename = std::move(other.wav_filename);

    // Reset the state of 'other'
    other.dataSize = 0;
    other.wav_filename = "";
  }
  return *this;
}

bool wav_writer::open(const std::string& filename, const uint32_t sample_rate,
                      const uint16_t bits_per_sample, const uint16_t channels) {
  if (open_wav(filename)) {
    return write_header(sample_rate, bits_per_sample, channels);
  }
  return false;
}

bool wav_writer::write(const int16_t* data, size_t length) {
  return write_audio(data, length);
}

bool wav_writer::close() {
  if (file.is_open()) {
    file.close();
  }
  return true;
}

bool wav_writer::write_header(const uint32_t sample_rate,
                              const uint16_t bits_per_sample,
                              const uint16_t channels) {
  file.write("RIFF", 4);
  file.write("\0\0\0\0", 4);  // Placeholder for file size
  file.write("WAVE", 4);
  file.write("fmt ", 4);

  const uint32_t sub_chunk_size = 16;
  const uint16_t audio_format = 1;  // PCM format
  const uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
  const uint16_t block_align = channels * bits_per_sample / 8;

  file.write(reinterpret_cast<const char*>(&sub_chunk_size), 4);
  file.write(reinterpret_cast<const char*>(&audio_format), 2);
  file.write(reinterpret_cast<const char*>(&channels), 2);
  file.write(reinterpret_cast<const char*>(&sample_rate), 4);
  file.write(reinterpret_cast<const char*>(&byte_rate), 4);
  file.write(reinterpret_cast<const char*>(&block_align), 2);
  file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
  file.write("data", 4);
  file.write("\0\0\0\0", 4);  // Placeholder for data size

  return true;
}

bool wav_writer::write_audio(const int16_t* data, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    const int16_t intSample = data[i];
    file.write(reinterpret_cast<const char*>(&intSample), sizeof(int16_t));
    dataSize += sizeof(int16_t);
  }
  if (file.is_open()) {
    file.seekp(4, std::ios::beg);
    uint32_t fileSize = 36 + dataSize;
    file.write(reinterpret_cast<char*>(&fileSize), 4);
    file.seekp(40, std::ios::beg);
    file.write(reinterpret_cast<char*>(&dataSize), 4);
    file.seekp(0, std::ios::end);
  }
  return true;
}

bool wav_writer::open_wav(const std::string& filename) {
  if (filename != wav_filename) {
    if (file.is_open()) {
      file.close();
    }
  }
  if (!file.is_open()) {
    file.open(filename, std::ios::binary);
    wav_filename = filename;
    dataSize = 0;
  }
  return file.is_open();
}