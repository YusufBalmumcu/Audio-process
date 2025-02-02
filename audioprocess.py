import wave
import pyaudio
import keyboard
import time

#  Audio Recording Constants
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono (set to 2 if stereo is needed)
CHUNK = 1024  # Buffer size
RATE = 44100  # Sample rate
OUTPUT_FILENAME = "output.wav"  # Output file

#  Select the correct input device
input_device_index = 0

#  Initialize PyAudio
audio = pyaudio.PyAudio()

#  Display audio devices
print("\nAvailable Audio Devices:")
for i in range(audio.get_device_count()):
    dev = audio.get_device_info_by_index(i)
    print(f"{i}: {dev['name']} | Channels: {dev['maxInputChannels']} | Rate: {int(dev['defaultSampleRate'])}")




if input_device_index is None:
    input_device_index = int(input("\nEnter the index of your input device: "))

#  Open the audio stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)

frames = []


print("🎤 Now Recording... Press SPACE to stop.")
def record():
    try:
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)  # Prevent buffer errors
                frames.append(data)
            except OSError as e:
                print(f"⚠️ Buffer Overflow Error: {e}")  # If buffer overflow happens, warn the user

            if keyboard.is_pressed("space"):  # Stop when SPACE is pressed
                print("🛑 Recording stopped.")
                time.sleep(0.3)  # Prevent double presses
                break
    except KeyboardInterrupt:
        print("🛑 Recording stopped.")
    # Stop & close the stream 
    stream.stop_stream()
    stream.close()
    audio.terminate()

record()

# 💾 Save to WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as waveFile:
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))

print(f"✅ Audio saved as {OUTPUT_FILENAME}")
