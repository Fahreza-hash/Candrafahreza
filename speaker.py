from gtts import gTTS # type: ignore
import tempfile
import os
import threading
import queue
import pygame
import time

class Speaker:
    def __init__(self):
        self.q = queue.Queue()
        self._running = True
        self._playing = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            text = self.q.get()
            if text is None:  # tandas berhenti
                break
            try:
                self._speak_once(text)
            except Exception as e:
                print(f"[ERROR SPEAK] {e}")
            self.q.task_done()

    def say(self, text: str):
        if text and text.strip():
            self.q.put(text)

    def _speak_once(self, text):
        if not self._running:
            return
        print(f"[SPEAK] {text}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts = gTTS(text=text, lang="id")
        tts.save(tmp.name)
        tmp.close()

        pygame.mixer.init()
        pygame.mixer.music.load(tmp.name)
        pygame.mixer.music.play()
        self._playing = True
        # Tunggu tapi tetap bisa diinterupsimefwasb v gw

        while pygame.mixer.music.get_busy() and self._running:
            time.sleep(0.1)

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        os.remove(tmp.name)
        self._playing = False

    def stop(self):
        """Berhentiin suara & thread"""
        print("[ STOP] Menghentikan speaker...")
        self._running = False
        if self._playing:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        self.q.put(None)
        self._thread.join(timeout=2)
        print("[ STOP] Speaker dimatikan.")
