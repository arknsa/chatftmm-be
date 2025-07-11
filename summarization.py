# ===================================================================
# File: summarization.py (MODIFIKASI)
# Perubahan: Menambahkan logika coba lagi (retry) untuk menangani error 429.
# ===================================================================
import requests
import json
import time
from config import settings

class Summarizer:
    def summarize(self, query: str, contexts: list[str]) -> str:
        combined_context = "\n\n---\n\n".join(contexts)
        prompt = f"""
        Anda adalah asisten AI dari Fakultas Teknologi Maju dan Multidisiplin (FTMM) Universitas Airlangga.
        Tugas Anda adalah menjawab pertanyaan pengguna secara akurat dan informatif HANYA berdasarkan informasi dari "Konteks Dokumen" yang diberikan.

        Ikuti langkah-langkah berpikir berikut untuk menyusun jawaban Anda:

        **Langkah 1: Analisis & Ekstraksi Fakta.**
        - Baca pertanyaan pengguna dengan saksama: "{query}"
        - Baca setiap potongan informasi di dalam "Konteks Dokumen" di bawah ini.
        - Dari konteks tersebut, identifikasi dan catat semua fakta, poin kunci, nama, tanggal, atau data yang secara langsung relevan dengan pertanyaan pengguna.
        - Abaikan informasi apa pun dalam konteks yang tidak berhubungan dengan pertanyaan.

        **Langkah 2: Sintesis & Penulisan Jawaban.**
        - Berdasarkan semua fakta relevan yang telah Anda ekstrak pada Langkah 1, susunlah sebuah jawaban yang koheren, jelas, dan mudah dipahami dalam Bahasa Indonesia.
        - Jawaban akhir harus secara langsung menjawab pertanyaan pengguna dan tidak boleh mengandung informasi dari luar konteks yang diberikan.
        - Jika setelah menganalisis semua konteks tidak ada informasi yang cukup untuk menjawab pertanyaan, nyatakan dengan sopan bahwa Anda tidak memiliki informasi yang cukup.
        - Jangan menyebutkan proses berpikir Anda atau "Langkah 1 / Langkah 2" dalam jawaban akhir.

        **Konteks Dokumen:**
        ---
        {combined_context}
        ---

        **Pertanyaan Pengguna:**
        "{query}"

        Sekarang, berikan jawaban akhir yang sudah Anda sintesis.
        """
        
        # --- PERUBAHAN DI SINI: LOGIKA RETRY ---
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": settings.HTTP_REFERER, 
                        "X-Title": "Chatbot FTMM"
                    },
                    data=json.dumps({
                        "model": "qwen/qwen-2.5-72b-instruct:free",
                        "messages": [
                            {"role": "system", "content": "Anda adalah asisten AI yang ahli dalam menganalisis teks dan merangkum informasi secara akurat."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.5
                    }),
                    timeout=60
                )

                if response.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    print(f"Rate limit terdeteksi. Menunggu {wait_time} detik...")
                    time.sleep(wait_time)
                    continue # Coba lagi

                response.raise_for_status()
                response_data = response.json()
                return response_data['choices'][0]['message']['content'].strip()
            
            except requests.exceptions.RequestException as e:
                print(f"Error dalam pemanggilan API OpenRouter (Percobaan {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5) # Jeda singkat sebelum retry
                else:
                    return "Maaf, terjadi kesalahan koneksi ke server AI."
            except (KeyError, IndexError) as e:
                print(f"Error parsing respons dari OpenRouter: {e}")
                return "Maaf, terjadi kesalahan dalam memproses respons dari server."
        
        return "Maaf, terjadi kesalahan setelah beberapa kali percobaan."
        # --- AKHIR PERUBAHAN ---
