# speech_synthesis.py
import pyttsx3
import time
import threading


# OPTION 1: Using pyttsx3 with better settings
class Speaker:
    def __init__(self, engine_type="gtts"):
        """
        Initialize speech engine.
        engine_type options:
        - "pyttsx3": Offline, fast, but robotic voice
        - "gtts": Google TTS, needs internet, better quality
        """
        self.engine_type = engine_type

        if engine_type == "pyttsx3":
            self._init_pyttsx3()
        elif engine_type == "gtts":
            self._init_gtts()

        self.last_message_time = 0
        self.last_critical_time = 0
        self.message_queue = []
        self.lock = threading.Lock()
        self.tts_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.tts_thread.start()

    def _init_pyttsx3(self):
        """Initialize pyttsx3 with optimized settings for Portuguese"""
        self.engine = pyttsx3.init()

        # List all available voices to find Portuguese
        voices = self.engine.getProperty("voices")

        print("\n[TTS] Vozes disponíveis:")
        pt_voice_found = False
        for i, voice in enumerate(voices):
            print(f"  {i}: {voice.name} | ID: {voice.id}")
            # Try to find Portuguese-BR voice
            if any(
                keyword in voice.id.lower()
                for keyword in ["portuguese", "brazil", "pt-br", "pt_br"]
            ):
                print(f"  ✓ Voz em Português encontrada!")
                self.engine.setProperty("voice", voice.id)
                pt_voice_found = True
                break

        if not pt_voice_found:
            print("[TTS] Nenhuma voz em português encontrada, usando padrão.")
            # On Linux, try to use espeak's Portuguese variant
            if len(voices) > 0:
                # Try voice index that might be Portuguese (varies by system)
                self.engine.setProperty("voice", voices[0].id)

        # Optimize speech parameters
        self.engine.setProperty("rate", 150)  # Speed (default is ~200)
        self.engine.setProperty("volume", 1.0)  # Volume (0.0 to 1.0)

    def _init_gtts(self):
        """Initialize Google TTS (requires internet)"""
        try:
            from gtts import gTTS
            import pygame
            import io

            self.gTTS = gTTS
            self.pygame = pygame
            pygame.mixer.init()
            self.io = io
            print("[TTS] Google TTS inicializado (requer internet)")
        except ImportError:
            print("[TTS] ERRO: gtts ou pygame não instalado.")
            print("      Instale com: pip install gtts pygame")
            print("      Voltando para pyttsx3...")
            self.engine_type = "pyttsx3"
            self._init_pyttsx3()

    def _process_queue(self):
        """Process speech queue in separate thread - NON-BLOCKING"""
        while True:
            with self.lock:
                if self.message_queue:
                    message, is_critical = self.message_queue.pop(0)

                    # Release lock BEFORE speaking to avoid blocking main thread
                    should_speak = True
                else:
                    should_speak = False

            if should_speak:
                if self.engine_type == "pyttsx3":
                    # This runs in the thread, not blocking main loop
                    self.engine.say(message)
                    self.engine.runAndWait()
                elif self.engine_type == "gtts":
                    self._speak_gtts(message)
            else:
                time.sleep(0.1)  # Small sleep when queue is empty

    def _speak_gtts(self, message):
        """Use Google TTS to speak (better quality)"""
        try:
            # Generate speech
            tts = self.gTTS(text=message, lang="pt-br", slow=False)

            # Save to memory buffer
            fp = self.io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            # Play audio
            self.pygame.mixer.music.load(fp)
            self.pygame.mixer.music.play()

            # Wait for playback to finish
            while self.pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            print(f"[TTS] Erro no Google TTS: {e}")

    def speak(self, message, is_critical=False, rate_limit_seconds=3):
        """
        Adiciona uma mensagem à fila de fala com controle de frequência.
        Alertas críticos têm prioridade mas TAMBÉM respeitam rate limiting.
        """
        current_time = time.time()

        # Critical messages have their own rate limiting (1 second)
        if is_critical:
            if current_time - self.last_critical_time > 1.0:
                with self.lock:
                    self.message_queue.clear()
                    self.message_queue.insert(0, (message, True))
                self.last_critical_time = current_time
            return

        # Normal messages respect the rate-limiting
        if current_time - self.last_message_time > rate_limit_seconds:
            with self.lock:
                if not self.message_queue:
                    self.message_queue.append((message, False))
            self.last_message_time = current_time


# RECOMMENDED USAGE:
# 1. Try pyttsx3 first (faster, offline): Speaker(engine_type="pyttsx3")
# 2. If voice quality is bad, use gtts (needs internet): Speaker(engine_type="gtts")

# For gtts, install dependencies:
# pip install gtts pygame
