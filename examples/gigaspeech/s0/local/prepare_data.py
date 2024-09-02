import os
import sys
import random

def prepare_data(transcript_file, output_dir, split_ratio=0.9):
    with open(transcript_file, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_point = int(len(lines) * split_ratio)
    train_lines = lines[:split_point]
    dev_lines = lines[split_point:]

    def write_files(lines, prefix):
        wav_scp = []
        text = []
        data_list = []

        for line in lines:
            filename, transcript = line.strip().split('|')
            wav_path = os.path.join(os.path.dirname(transcript_file), filename + '.wav')
            
            if not os.path.exists(wav_path):
                print(f"Warning: {wav_path} does not exist. Skipping.")
                continue

            wav_scp.append(f"{filename} {wav_path}")
            text.append(f"{filename} {transcript}")
            data_list.append(f'{{"key": "{filename}", "wav": "{wav_path}", "txt": "{transcript}"}}')

        with open(os.path.join(output_dir, f'{prefix}_wav.scp'), 'w') as f:
            f.write('\n'.join(wav_scp))
        with open(os.path.join(output_dir, f'{prefix}_text'), 'w') as f:
            f.write('\n'.join(text))
        with open(os.path.join(output_dir, f'{prefix}_data.list'), 'w') as f:
            f.write('\n'.join(data_list))

    write_files(train_lines, 'train')
    write_files(dev_lines, 'dev')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_data.py /path/to/transcript.txt /path/to/output/directory")
        sys.exit(1)
    
    transcript_file = sys.argv[1]
    output_dir = sys.argv[2]
    prepare_data(transcript_file, output_dir)