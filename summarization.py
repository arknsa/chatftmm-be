import time
from openai import RateLimitError, APIConnectionError, APIStatusError
import model_service
from typing import List, Dict, Any, Optional

class Summarizer:
    # --- MODIFIED: Updated signature to accept history ---
    def summarize(self, query: str, contexts: list[str], history: Optional[List[Dict[str, Any]]] = None) -> str:
        client = model_service.openai_client
        if not client:
            return "Maaf, koneksi ke layanan AI tidak siap. Periksa log server."

        combined_context = "\n\n---\n\n".join(contexts)

        # --- NEW: Build the conversation history string ---
        history_str = ""
        if history:
            for message in history:
                role = "Pengguna" if message.get('role') == 'user' else "Asisten"
                history_str += f"{role}: {message.get('content')}\n"
        
        # --- MODIFIED: The prompt now includes the conversation history ---
        prompt = f"""
Anda adalah asisten AI dari Fakultas Teknologi Maju dan Multidisiplin (FTMM) Universitas Airlangga.
Anda harus menjawab dengan **SANGAT AKURAT, ringkas, dan profesional** berdasarkan `KONTEKS` yang diberikan dan `RIWAYAT PERCAKAPAN` sebelumnya. Ikuti semua peraturan di bawah ini.

**PERATURAN WAJIB (HARUS DIIKUTI):**

1.  **ATURAN PALING PENTING DAN TIDAK BOLEH DILANGGAR:**
    * Singkatan **"TI"** dalam konteks FTMM **SELALU DAN HANYA** berarti **"Teknik Industri"**.
    * **DILARANG KERAS** menjawab atau berasumsi bahwa "TI" adalah "Teknologi Informasi".
    * Jika Anda melanggar aturan ini, jawaban Anda dianggap gagal total. Ulangi: **TI ADALAH TEKNIK INDUSTRI.**

2.  **KONTEKS PERCAKAPAN**: Gunakan `RIWAYAT PERCAKAPAN` untuk memahami pertanyaan lanjutan. Contoh: Jika pengguna bertanya "bagaimana dengan syaratnya?" setelah bertanya tentang beasiswa, Anda harus paham "syaratnya" merujuk pada beasiswa.

3.  **SUMBER TUNGGAL**: Jawaban HARUS 100% berdasarkan informasi dari `KONTEKS`. DILARANG KERAS menggunakan pengetahuan eksternal.

4.  **ATURAN SPESIFIK LAINNYA**:
    * **ATURAN DEKAN**: Jika ditanya tentang **"Dekan"**, bedakan dengan jelas antara **Plt. Dekan saat ini (Prof. Ni'matuzahroh)** dan **Dekan terdahulu (Prof. Dwi Setyawan)**. Jangan gabungkan informasi mereka.
    * Jika diminta data seperti UKT atau daftar lainnya, **selalu gunakan format tabel Markdown** agar mudah dibaca.

5.  **FORMAT JAWABAN**:
    * Gunakan Markdown (`**teks tebal**`, daftar, tabel) untuk membuat jawaban mudah dibaca.
    * **Langsung ke Inti Jawaban**: Hindari kalimat pembuka yang tidak perlu.

6.  **KEJUJURAN MUTLAK**: Jika informasi yang diminta pengguna TIDAK ADA di dalam `KONTEKS`, jawab HANYA dengan kalimat: **"Maaf, saya tidak dapat menemukan informasi spesifik mengenai hal tersebut."**

---
**RIWAYAT PERCAKAPAN (untuk konteks):**
{history_str}
---
**KONTEKS YANG RELEVAN (sumber utama jawaban):**
{combined_context}
---

**PERTANYAAN PENGGUNA SAAT INI:**
"{query}"

**JAWABAN AKURAT, TERFORMAT, DAN PATUH PADA SEMUA ATURAN DI ATAS:**
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Percobaan ke-{attempt + 1} untuk menghubungi OpenAI API...")
                
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.0
                )
                
                content = response.choices[0].message.content
                if content:
                    print("Respons dari OpenAI API berhasil diterima.")
                    return content.strip()
                else:
                    return "Maaf, AI memberikan respons kosong."

            except RateLimitError as e:
                wait_time = (2 ** attempt) * 5
                print(f"Terkena rate limit. Mencoba lagi dalam {wait_time} detik... Error: {e}")
                time.sleep(wait_time)
            
            except (APIConnectionError, APIStatusError) as e:
                print(f"Error koneksi atau status dari OpenAI API: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return "Maaf, terjadi masalah koneksi saat menghubungi server OpenAI."
            
            except Exception as e:
                print(f"Terjadi error tak terduga: {e}")
                return "Maaf, terjadi kesalahan internal yang tidak terduga."

        return "Maaf, saya tidak dapat memproses permintaan Anda setelah beberapa kali percobaan."

