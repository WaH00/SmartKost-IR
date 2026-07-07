import google.generativeai as genai
import os

# Ganti pakai API Key lu yang 'AQ.' tadi
genai.configure(api_key="MASUKKAN_GCP_API_KEY_DISINI")

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"MODEL YANG BISA DIPAKAI: {m.name}")