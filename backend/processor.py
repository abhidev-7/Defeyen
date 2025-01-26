import pyaudio
import websocket
import json
import boto3
import threading
import mss
import numpy as np
from PIL import Image
import io
import time
import os

# AWS Rekognition client
rekognition_client = boto3.client('rekognition')

# Function to capture frames from a specific screen region
def capture_screen(region):
    with mss.mss() as sct:
        while True:
            # Capture the region of the screen
            screenshot = sct.grab(region)
            # Convert screenshot to a PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            yield img
            time.sleep(1 / 5)  # Capture at ~5 FPS

# Function to process an image frame using AWS Rekognition
def process_frame_with_rekognition(image):
    # Convert the PIL Image to bytes
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        image_bytes = output.getvalue()

    # Detect objects using Rekognition
    response = rekognition_client.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=10,
        MinConfidence=75
    )

    # Print detected labels
    print("Rekognition Response:")
    for label in response['Labels']:
        print(f"Label: {label['Name']}, Confidence: {label['Confidence']:.2f}%")

# Function to capture audio and stream it to GPT-4o
def audio_stream_to_gpt4o(audio_device_index, gpt4o_ws_url, api_key):
    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=audio_device_index,
        frames_per_buffer=1024
    )

    # WebSocket connection to GPT-4o API
    def on_message(ws, message):
        print("GPT-4o Response:", message)

    def on_error(ws, error):
        print("Error:", error)

    def on_close(ws, close_status_code, close_msg):
        print("WebSocket Closed")

    def on_open(ws):
        print("WebSocket Opened")
        while True:
            data = stream.read(1024)
            ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)

    ws = websocket.WebSocketApp(
        gpt4o_ws_url,
        header={"Authorization": f"Bearer {api_key}"},
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.start()

# Main function to coordinate audio and visual processing
def main():
    # GPT-4o WebSocket URL and API Key
    gpt4o_ws_url = "https://api.openai.com/v1/realtime"  # Replace with actual endpoint
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # List audio devices to find the virtual audio cable
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(f"Device {i}: {p.get_device_info_by_index(i)['name']}")

    audio_device_index = int(input("Enter the device index for the virtual audio cable: "))

    # Start the audio processing thread
    threading.Thread(target=audio_stream_to_gpt4o, args=(audio_device_index, gpt4o_ws_url, api_key)).start()

    # Define the screen region to capture (adjust based on WhatsApp call location)
    region = {"left": 100, "top": 100, "width": 800, "height": 600}  # Adjust these values as needed

    # Start capturing and processing frames
    for frame in capture_screen(region):
        process_frame_with_rekognition(frame)

if __name__ == "__main__":
    main()
