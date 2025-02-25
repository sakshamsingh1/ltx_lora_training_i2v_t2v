from analyse_vae_audio_auffusion import VAEAudioAnalyse

vae_obj = VAEAudioAnalyse()
audio_path = '/home/sxk230060/TI2AV/misc/ltx_lora_training_i2v_t2v/outputs/temp/cat_sample.wav'
latent = vae_obj.audio_to_latent(audio_path)
save_path = '/home/sxk230060/TI2AV/misc/ltx_lora_training_i2v_t2v/outputs/temp/cat_reconstructed.wav'
vae_obj.latent_to_audio(latent, save_path)