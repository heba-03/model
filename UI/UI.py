import tkinter as tk
from tkinter import filedialog, ttk
import customtkinter as ctk
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch

# Load models and processor outside the function to avoid reloading each time
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

emotion_model_name = "r-f/wav2vec-english-speech-emotion-recognition"
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(emotion_model_name)


# Function to process audio, transcribe text, and detect emotion
def speech_to_text(audio_file):
    speech, sr = librosa.load(audio_file, sr=16000)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    speech_tensor = torch.from_numpy(speech).unsqueeze(0)

    with torch.no_grad():
        logits = model(**inputs).logits
        emotion_logits = emotion_model(speech_tensor).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    predicted_emotion_ids = torch.argmax(emotion_logits, dim=-1)
    emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

    predicted_emotion = emotion_labels[
        predicted_emotion_ids[0].item()] if predicted_emotion_ids.numel() > 0 else "No emotion detected."

    return transcription, predicted_emotion

def apply_mode(mode):
    if mode == "Dark":
        bg_color = "#121212"
        fg_color = "#ffffff"
        result_bg = "#1e1e1e"
        frame_color = "#2b2b2b"
    else:
        bg_color = "white"
        fg_color = "black"
        result_bg = "lightgrey"
        frame_color = "white"

    root.configure(bg=bg_color)
    frame1.configure(bg=frame_color)
    frame2.configure(bg=frame_color)
    frame3.configure(bg=frame_color)

    mode_label.config(bg=frame_color, fg=fg_color)
    font_label.config(bg=frame_color, fg=fg_color)
    title_label.config(bg=frame_color, fg=fg_color)
    result_label.config(bg=result_bg, fg=fg_color)


    mode_dropdown.config(background=frame_color, foreground=fg_color)
    font_dropdown.config(background=frame_color, foreground=fg_color)

def apply_font_size(size_percentage):
    base_font_size = 12
    size = int(base_font_size * (int(size_percentage.strip('%')) / 100))
    result_label.config(font=("Arial", size))
    mode_label.config(font=("Arial", size))
    font_label.config(font=("Arial", size))
    title_label.config(font=("Arial", size))

root = tk.Tk()
root.title("Speech Emotion Detection")
root.geometry("800x400")
root.configure(bg="white")

frame1 = tk.Frame(root, bd=2, relief="groove", bg="white")
frame1.pack(side="left", fill=tk.BOTH, expand=True, padx=10, pady=10)

mode_label = tk.Label(frame1, text="Page Mode:", bg="white", fg="black")
mode_label.grid(row=0, column=0, padx=5, pady=10, sticky="e")

mode_var = tk.StringVar()
mode_var.set("Light")
mode_dropdown = ttk.Combobox(frame1, textvariable=mode_var)
mode_dropdown['values'] = ('Light', 'Dark')
mode_dropdown.grid(row=0, column=1, padx=5, pady=10)
mode_dropdown.bind("<<ComboboxSelected>>", lambda e: apply_mode(mode_var.get()))

font_label = tk.Label(frame1, text="Font Size:", bg="white", fg="black")
font_label.grid(row=1, column=0, padx=5, pady=10, sticky="e")

font_var = tk.StringVar()
font_var.set("100%")
font_dropdown = ttk.Combobox(frame1, textvariable=font_var)
font_dropdown['values'] = ('50%', '75%', '100%', '125%', '150%')
font_dropdown.grid(row=1, column=1, padx=5, pady=10)
font_dropdown.bind("<<ComboboxSelected>>", lambda e: apply_font_size(font_var.get()))

frame2 = tk.Frame(root, bd=2, relief="groove", bg="white")
frame2.pack(side="left", fill=tk.BOTH, expand=True, padx=10, pady=10)

title_label = tk.Label(frame2, text="Insert an Audio", font=("Arial", 16, "bold"), bg="white")
title_label.pack(pady=(10, 5))

def upload_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        transcription, emotion = speech_to_text(file_path)
        result_text.set(f"Transcribed Text: {transcription}\nDetected Emotion: {emotion}")

upload_button = ctk.CTkButton(master=frame2, text="Upload Audio", command=upload_audio)
upload_button.pack(pady=5)

frame3 = tk.Frame(root, bd=2, relief="groove", bg="white")
frame3.pack(side="left", fill=tk.BOTH, expand=True, padx=10, pady=10)

result_text = tk.StringVar(value="Transcribed Text and Detected Emotion will appear here.")
result_label = tk.Label(frame3, textvariable=result_text, wraplength=200, justify="center",
                        bg="lightgrey", highlightbackground="darkgrey", highlightthickness=10)
result_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()
