import json
import whisper # Main ASR model
# `pyspellchecker` is imported conditionally within a method.

# Attempt to get Whisper's language list at the module level for INPUT_TYPES
# This helps in scenarios where the class might be inspected before instantiation.
try:
    _INITIAL_WHISPER_LANGUAGES = whisper.tokenizer.LANGUAGES
except (NameError, AttributeError, ImportError):
    # Fallback if whisper or its tokenizer isn't available/imported when this module is first read.
    # This can happen depending on the execution environment or import order.
    _INITIAL_WHISPER_LANGUAGES = {"en": "english"} # Default to prevent errors
    # A warning could be logged here if a logging mechanism is available.
    print("Initial Warning: Whisper module or tokenizer.LANGUAGES not fully available during module load. Language list may be incomplete until class initialization or INPUT_TYPES call.")

class CobotWhisperToTransciption:
    # Class-level attributes for Whisper language options.
    # These are populated/updated by INPUT_TYPES or __init__ to ensure they are current.
    _whisper_tokenizer_languages_cache = _INITIAL_WHISPER_LANGUAGES
    WHISPER_LANGUAGES_SUPPORTED = ["auto"] + \
        [lang.capitalize() for lang in sorted(list(_whisper_tokenizer_languages_cache.values()))]
    _WHISPER_LANGUAGE_NAME_TO_CODE_CACHE = {
        name.lower(): code for code, name in _whisper_tokenizer_languages_cache.items()
    }

    def __init__(self):
        # Ensure language maps are up-to-date if whisper was loaded after initial module parsing.
        # This is a good place for a one-time setup if needed, though INPUT_TYPES also handles it.
        self._refresh_whisper_language_data()

    @classmethod
    def _refresh_whisper_language_data(cls):
        """Ensures Whisper language data is current."""
        try:
            current_langs = whisper.tokenizer.LANGUAGES
            if cls._whisper_tokenizer_languages_cache != current_langs:
                cls._whisper_tokenizer_languages_cache = current_langs
                cls.WHISPER_LANGUAGES_SUPPORTED = ["auto"] + \
                    [lang.capitalize() for lang in sorted(list(current_langs.values()))]
                cls._WHISPER_LANGUAGE_NAME_TO_CODE_CACHE = {
                    name.lower(): code for code, name in current_langs.items()
                }
        except (NameError, AttributeError, ImportError) as e:
            print(f"Warning: Could not refresh Whisper language data: {e}")
            # Keep using cached/fallback data

    @classmethod
    def INPUT_TYPES(cls):
        cls._refresh_whisper_language_data() # Ensure language list is fresh

        spell_check_options = ["None", "English", "Spanish", "French",
                               "Portuguese", "German", "Italian",
                               "Russian", "Arabic", "Basque", "Latvian", "Dutch"]
        transcription_mode_options = ["word", "line", "fill"]
        # Common Whisper models. Add more if needed (e.g., tiny.en, base.en)
        whisper_model_options = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]

        return {
            "required": {
                "audio_file": ("STRING", {"display": "text", "forceInput": True, "multiline": False}),
                "whisper_model": (whisper_model_options, {"default": "base", "display": "dropdown"}),
                "whisper_language": (cls.WHISPER_LANGUAGES_SUPPORTED, {"default": "auto", "display": "dropdown"}),
                "spell_check_language": (spell_check_options, {"default": "None", "display": "dropdown"}),
                "framestamps_max_chars": ("INT", {"default": 40, "min":1, "step": 1, "display": "number"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1, "display": "number"}),
                "transcription_mode": (transcription_mode_options, {"default": "fill", "display": "dropdown"}),
                "uppercase": ("BOOLEAN", {"default": True, "display": "toggle"})
            }
        }

    CATEGORY = "CObot" # Kept original category, added Audio subcategory
    RETURN_TYPES = ("TRANSCRIPTION", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("transcription_data_dict", "text_output", "framestamped_text", "json_timestamps",)
    FUNCTION = "execute_transcription" # Changed from "run" to be more descriptive
    OUTPUT_NODE = True # Indicates this node produces a final output

    def execute_transcription(self, audio_file, whisper_model, whisper_language, spell_check_language, framestamps_max_chars, **kwargs):
        fps = kwargs.get('fps', 30)
        transcription_mode_setting = kwargs.get('transcription_mode', "fill") # Renamed from transcription_mode to avoid conflict
        should_uppercase = kwargs.get('uppercase', True)

        print(f"Starting transcription for: {audio_file}")
        print(f"Using Whisper model: {whisper_model}, Language: {whisper_language}")
        print(f"Spell check: {spell_check_language}, FPS for framestamps: {fps}, Max chars/line: {framestamps_max_chars}")
        print(f"Uppercase: {should_uppercase}, Transcription mode setting: {transcription_mode_setting}")


        # Load Whisper model
        try:
            model = whisper.load_model(whisper_model)
            print(f"Whisper model '{whisper_model}' loaded.")
        except Exception as e:
            print(f"Error loading Whisper model '{whisper_model}': {e}")
            # Return empty/error state for all outputs
            error_settings_dict = {"transcription_data": [], "fps": fps, "transcription_mode": transcription_mode_setting}
            return (error_settings_dict, f"ERROR: Could not load model {whisper_model}", "", "[]")

        # Prepare transcription arguments for Whisper
        transcribe_args = {}
        if whisper_language.lower() != "auto":
            # Use the cached map from the class
            lang_code = CobotWhisperToTransciption._WHISPER_LANGUAGE_NAME_TO_CODE_CACHE.get(whisper_language.lower())
            if lang_code:
                transcribe_args['language'] = lang_code
                print(f"Whisper transcription language set to: {whisper_language} (code: {lang_code})")
            else:
                print(f"Warning: Whisper language '{whisper_language}' not recognized. Using auto-detection.")
        else:
            print("Whisper will auto-detect language.")
            
        # Transcribe with Whisper
        try:
            print("Starting audio transcription with Whisper...")
            result = model.transcribe(audio_file, word_timestamps=True, **transcribe_args)
            print("Whisper transcription complete.")
        except FileNotFoundError:
            print(f"ERROR: Audio file not found at path: {audio_file}")
            error_settings_dict = {"transcription_data": [], "fps": fps, "transcription_mode": transcription_mode_setting}
            return (error_settings_dict, f"ERROR: Audio file not found - {audio_file}", "", "[]")
        except Exception as e:
            print(f"Error during Whisper transcription: {e}")
            error_settings_dict = {"transcription_data": [], "fps": fps, "transcription_mode": transcription_mode_setting}
            return (error_settings_dict, f"ERROR: Transcription failed - {e}", "", "[]")

        # Convert Whisper's word timestamps to the expected format: [(word, start_time, end_time), ...]
        raw_transcription_tuples = []
        if 'segments' in result:
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment["words"]:
                        word_text = word_info.get("word", "").strip() # Ensure 'word' key and strip spaces
                        start_time = word_info.get("start")
                        end_time = word_info.get("end")
                        
                        if word_text and start_time is not None and end_time is not None:
                            raw_transcription_tuples.append((word_text, float(start_time), float(end_time)))
                        # else:
                        #     print(f"Debug: Skipping word due to missing data: {word_info}") # Optional debug
        
        if not raw_transcription_tuples and result.get("text"):
             print("Warning: No word-level timestamps extracted. The output might be based on full text only.")
             # If needed, one could attempt to split result["text"] and assign dummy/segment timings.
             # For now, if no words, subsequent processing will handle empty lists gracefully.

        # Correct with spell checker (if a language is selected)
        if spell_check_language.lower() != "none":
            corrected_transcription_intermediate = self._correct_transcription_spellchecker(raw_transcription_tuples, spell_check_language)
        else:
            print("Spell checking skipped as per settings.")
            corrected_transcription_intermediate = raw_transcription_tuples

        # Apply casing
        processed_transcription_data = []
        for word, start_time, end_time in corrected_transcription_intermediate:
            processed_word = word.upper() if should_uppercase else word.lower()
            processed_transcription_data.append((processed_word, start_time, end_time))
        
        # Generate string formatted for framestamps
        framestamped_text_output = self._format_transcription_to_framestamps(processed_transcription_data, fps, framestamps_max_chars)
        
        # Convert processed transcription to a single raw string
        text_output_string = self._format_transcription_to_plain_string(processed_transcription_data)
        
        # Convert processed transcription to JSON string with timestamps
        json_timestamps_output = self._format_transcription_to_json_timestamps(processed_transcription_data)
        
        # This is the "TRANSCRIPTION" type output, a dictionary
        transcription_data_dictionary = {
            "transcription_data": processed_transcription_data, # List of (word, start, end)
            "fps": fps,
            "transcription_mode": transcription_mode_setting # User's selected mode for downstream use
        }
        print("Transcription processing finished.")
        return (transcription_data_dictionary, text_output_string, framestamped_text_output, json_timestamps_output,)

    def _format_transcription_to_plain_string(self, transcription_data_list):
        """Converts a list of (word, start, end) tuples to a space-separated string."""
        return ' '.join([word for word, _, _ in transcription_data_list])

    def _format_transcription_to_framestamps(self, transcription_data_list, fps, max_chars_per_line):
        """Formats transcription into a string with frame numbers and text lines, respecting max_chars_per_line."""
        output_lines = []
        current_line_buffer = ""
        current_line_start_frame = 0

        for word, start_time, _ in transcription_data_list:
            frame_number = round(start_time * fps)

            if not current_line_buffer:  # First word of a new line segment
                current_line_buffer = word
                current_line_start_frame = frame_number
            elif len(current_line_buffer + " " + word) <= max_chars_per_line:
                current_line_buffer += " " + word
            else:
                # Current line_buffer is complete, add it to output
                output_lines.append(f'"{current_line_start_frame}": "{current_line_buffer.strip()}",')
                # Start a new line segment with the current word
                current_line_buffer = word
                current_line_start_frame = frame_number
        
        # Add any remaining content in current_line_buffer (the last line)
        if current_line_buffer:
            output_lines.append(f'"{current_line_start_frame}": "{current_line_buffer.strip()}",')
        
        # Join all formatted lines. Add a newline at the end if there's content.
        formatted_text = "\n".join(output_lines)
        if formatted_text and not formatted_text.endswith('\n'):
            formatted_text += '\n'
            
        return formatted_text

    def _format_transcription_to_json_timestamps(self, transcription_data_list):
        """Converts transcription data to a JSON string with word, start_time, and end_time."""
        json_data = [{"word": word, "start_time": round(start_time, 3), "end_time": round(end_time, 3)} 
                       for word, start_time, end_time in transcription_data_list]
        return json.dumps(json_data, indent=2)

    def _correct_transcription_spellchecker(self, raw_transcription_tuples, spell_check_language_full_name):
        """Applies spell correction to transcribed words if a supported language is chosen."""
        # Map full language names to ISO codes recognized by pyspellchecker
        language_code_map = {
            "English": "en", "Spanish": "es", "French": "fr",
            "Portuguese": "pt", "German": "de", "Italian": "it",
            "Russian": "ru", "Arabic": "ar", "Basque": "eu",
            "Latvian": "lv", "Dutch": "nl"
            # "None" is handled before calling this function
        }
            
        language_code = language_code_map.get(spell_check_language_full_name)
        if not language_code:
            print(f"Spell check language '{spell_check_language_full_name}' not supported by internal map. Skipping spell check.")
            return raw_transcription_tuples

        try:
            from spellchecker import SpellChecker
            spell = SpellChecker(language=language_code)
            print(f"Spell checking enabled for language: {spell_check_language_full_name} (code: {language_code})")
        except ImportError:
            print("Module 'pyspellchecker' not found. Please install it (pip install pyspellchecker) to use spell checking. Skipping.")
            return raw_transcription_tuples
        except Exception as e: 
            print(f"Error initializing SpellChecker for language '{language_code}': {e}. Skipping spell check.")
            return raw_transcription_tuples

        corrected_transcription = []
        for word, start_time, end_time in raw_transcription_tuples:
            if not word.strip(): # Skip empty or whitespace-only words
                corrected_transcription.append((word, start_time, end_time))
                continue

            # Attempt correction. Pyspellchecker is case-sensitive by default.
            # Forcing to lower for correction might be an an option if issues arise with mixed-case words.
            corrected_word_candidate = spell.correction(word)
            
            # Use original word if no correction, or if correction is None (e.g. for punctuation or unknown words)
            final_word = corrected_word_candidate if corrected_word_candidate is not None else word
            corrected_transcription.append((final_word, start_time, end_time))

        print("Spell checking process completed.")
        return corrected_transcription

# Example of how to register this node if it's for a specific framework like ComfyUI
# This part is illustrative and depends on the target framework.
# If this is a standalone script, this section would not be present.
#


if __name__ == '__main__':
    # This is a basic example for testing the class directly.
    # You would need to have an audio file (e.g., "test.wav") and Whisper installed.
    print("Running basic test for CobotWhisperToTransciption class...")

    # Create a dummy audio file path for the example
    dummy_audio_file = "test_audio.wav" # Replace with a real path to a .wav or .mp3 file
    # A simple way to create a silent wav for testing if you have `scipy` and `numpy`:
    # import numpy as np
    # from scipy.io.wavfile import write
    # samplerate = 16000
    # duration = 1 # seconds
    # frequency = 440 # Hz
    # t = np.linspace(0., duration, int(samplerate*duration), endpoint=False)
    # amplitude = np.iinfo(np.int16).max * 0.1
    # data = amplitude * np.sin(2. * np.pi * frequency * t)
    # write(dummy_audio_file, samplerate, data.astype(np.int16))
    # print(f"Created dummy audio file: {dummy_audio_file}")
    
    # Check if Whisper is available before running a full test that requires it
    try:
        import whisper
        whisper_available = True
    except ImportError:
        print("WARNING: openai-whisper is not installed. Full test cannot run.")
        whisper_available = False

    if whisper_available:
        # Instantiate the class
        s2t_node = CobotWhisperToTransciption()

        # Simulate input parameters (use a small model for faster testing)
        # Ensure 'dummy_audio_file' exists and is a valid audio file for a real test.
        # For this example, we'll rely on the FileNotFoundError handling if it doesn't exist.
        print(f"\nAttempting transcription of '{dummy_audio_file}' (if it exists)...")
        try:
            results = s2t_node.execute_transcription(
                audio_file=dummy_audio_file,
                whisper_model="tiny", # Use a small model for quick testing
                whisper_language="auto",
                spell_check_language="English", # or "None"
                framestamps_max_chars=50,
                fps=25,
                transcription_mode="fill",
                uppercase=False
            )

            transcription_dict, text_str, framestamps_str, json_str = results

            print("\n--- RESULTS ---")
            print("\n1. Transcription Data Dictionary (TRANSCRIPTION):")
            print(f"  FPS: {transcription_dict['fps']}")
            print(f"  Mode: {transcription_dict['transcription_mode']}")
            print(f"  Data (first 5 words): {transcription_dict['transcription_data'][:5]}")
            
            print("\n2. Plain Text Output (STRING):")
            print(text_str)

            print("\n3. Framestamped Text (STRING):")
            print(framestamps_str)

            print("\n4. JSON Timestamps (STRING):")
            print(json_str)

        except FileNotFoundError:
            print(f"\nTEST INFO: The dummy audio file '{dummy_audio_file}' was not found. "
                  "Create a real audio file at this path or update the path to test transcription.")
        except Exception as e:
            print(f"\nAn error occurred during the test run: {e}")
    else:
        print("Skipped transcription test as Whisper is not available.")

    print("\nBasic class structure test complete.")


NODE_CLASS_MAPPINGS = {
 "CobotWhisperToTransciption": CobotWhisperToTransciption
}
NODE_DISPLAY_NAME_MAPPINGS = {
 "CobotWhisperToTransciption": "CObot Whisper To Transciption"
}