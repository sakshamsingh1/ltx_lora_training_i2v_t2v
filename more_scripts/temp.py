from analyse_vae_audio_auffusion import VAEAudioAnalyse

vae_obj = VAEAudioAnalyse()
audio_path = '/mnt/ssd0/saksham/i2av/AVSync15/audios_together/I4JKaLOIEGs_000030_000040_3.0_6.0.wav'
latent = vae_obj.audio_to_latent(audio_path)
save_path = 'temp/reconstructed.wav'
vae_obj.latent_to_audio(latent, save_path)