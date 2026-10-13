# my_secrets.py

HF_TOKEN = ""       # Hugging Face token
GPT_API_KEY = ""
GROQ_KEY = ""  # Gorq API key
PROMPT = ("Perform QC checks for corruption and file extension, sample rate check, audio length, VAD silence detection, "
          "upsampling, number of speakers, speaker duration, verify if speakers are new or old, IndicLID language "
          "identification, verify language as Hindi, CTC score, WER computation, normalization, LLM score, check domain, "
          "English word count and utterance duplicate check in '/Users/pw/2une/speechQM/audios_qc/hindi/audios'")
    # user_prompt = "I want to check the number of speakers, calculate the duration for each speaker, verify if speakers are new or old, compute WER, count English words, check for utterance duplicates, and clean the transcriptions by removing HTML tags in the audio files located at '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/'."
    # user_prompt= "verify language as Hindi (' hi '),CTC SCORE,   WER computation , normalization , calculate LLM score , check domain ,english word counter, and check for utterance duplicate  in :'/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC2/audios/' "
    # user_prompt= "Perform QC Checks, Sample rate check, VAD silence detection, upsampling, IndicLID language identification, calculate speaker duration, verify if speakers are new or old in :'/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/' "
    # user_prompt = "I need to analyze my Hindi-English conversational dataset to assess the quality of dialogue flow, contextual consistency across speaker turns, and conversational coherence for training our dialogue state tracking model. The data contains multi-turn conversations where speakers ask questions and provide answers in alternating fashion. audios present in directory: '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC2/audios/' "
    
