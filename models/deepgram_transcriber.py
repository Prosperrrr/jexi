import os
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

class DeepgramTranscriber:
    """
    Real-time transcription using Deepgram API
    """
    
    def __init__(self, api_key=None):
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('DEEPGRAM_API_KEY')
        
        if not self.api_key:
            print("⚠️ Deepgram API key not found!")
            print("   Set DEEPGRAM_API_KEY environment variable or pass api_key parameter")
            self.client = None
        else:
            try:
                self.client = DeepgramClient(self.api_key)
                print("✅ Deepgram transcriber initialized!")
            except Exception as e:
                print(f"❌ Failed to initialize Deepgram: {e}")
                self.client = None
    
    def create_live_transcription(self, on_transcript, on_error=None):
        """
        Create a live transcription connection
        
        Args:
            on_transcript: Callback function(transcript_text)
            on_error: Optional error callback
            
        Returns:
            LiveTranscription connection object
        """
        if not self.client:
            print("❌ Deepgram client not initialized")
            return None
        
        try:
            # Create connection
            connection = self.client.listen.live.v("1")
            
            # Set up event handlers
            def handle_transcript(self, result, **kwargs):
                try:
                    sentence = result.channel.alternatives[0].transcript
                    if sentence:
                        on_transcript(sentence)
                except Exception as e:
                    print(f"Error handling transcript: {e}")
            
            def handle_error(self, error, **kwargs):
                if on_error:
                    on_error(error)
                else:
                    print(f"Deepgram error: {error}")
            
            # Register events
            connection.on(LiveTranscriptionEvents.Transcript, handle_transcript)
            connection.on(LiveTranscriptionEvents.Error, handle_error)
            
            # Options for transcription
            options = LiveOptions(
                model="nova-2",
                language="en",
                smart_format=True,
                interim_results=True,
                punctuate=True
            )
            
            # Start connection
            if connection.start(options):
                print("✅ Deepgram live transcription started")
                return connection
            else:
                print("❌ Failed to start Deepgram connection")
                return None
                
        except Exception as e:
            print(f"❌ Error creating live transcription: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_configured(self):
        """Check if Deepgram is properly configured"""
        return self.client is not None