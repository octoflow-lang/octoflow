//! Audio playback via raw Win32 MME API — zero external dependencies.
//!
//! Provides two builtins:
//! - `audio_play(samples, sample_rate)` — play PCM f32 samples (mono, 16-bit output)
//! - `audio_play_file(path)` — play a WAV file asynchronously

/// Play PCM audio samples via waveOut (blocking until complete).
/// `samples`: f32 array in [-1.0, 1.0] range.
/// `sample_rate`: e.g. 44100.
/// Returns 1.0 on success, 0.0 on failure.
pub fn audio_play_impl(samples: &[f32], sample_rate: u32) -> f32 {
    #[cfg(target_os = "windows")]
    {
        if samples.is_empty() || sample_rate == 0 {
            return 0.0;
        }

        use crate::platform::win32::*;

        // Guard: max ~10 min of audio at 48kHz to prevent overflow
        const MAX_SAMPLES: usize = 48_000 * 600; // ~29M samples
        let sample_count = samples.len().min(MAX_SAMPLES);

        // Convert f32 [-1.0, 1.0] → i16 PCM
        let mut pcm: Vec<i16> = Vec::with_capacity(sample_count);
        for &s in &samples[..sample_count] {
            let clamped = s.clamp(-1.0, 1.0);
            pcm.push((clamped * 32767.0) as i16);
        }

        // Safe byte length calculation (checked)
        let byte_len = match pcm.len().checked_mul(2) {
            Some(n) if n <= u32::MAX as usize => n as u32,
            _ => return 0.0,
        };

        let wfx = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM,
            nChannels: 1,
            nSamplesPerSec: sample_rate,
            nAvgBytesPerSec: sample_rate * 2, // 16-bit mono = 2 bytes per sample
            nBlockAlign: 2,
            wBitsPerSample: 16,
            cbSize: 0,
        };

        unsafe {
            let mut hwo: HWAVEOUT = std::ptr::null_mut();
            let result = waveOutOpen(
                &mut hwo, WAVE_MAPPER, &wfx,
                0, 0, CALLBACK_NULL,
            );
            if result != 0 {
                return 0.0;
            }

            let mut hdr = WAVEHDR {
                lpData: pcm.as_mut_ptr() as *mut i8,
                dwBufferLength: byte_len,
                dwBytesRecorded: 0,
                dwUser: 0,
                dwFlags: 0,
                dwLoops: 0,
                lpNext: std::ptr::null_mut(),
                reserved: 0,
            };

            let hdr_size = std::mem::size_of::<WAVEHDR>() as u32;
            if waveOutPrepareHeader(hwo, &mut hdr, hdr_size) != 0 {
                waveOutClose(hwo);
                return 0.0;
            }
            if waveOutWrite(hwo, &mut hdr, hdr_size) != 0 {
                waveOutUnprepareHeader(hwo, &mut hdr, hdr_size);
                waveOutClose(hwo);
                return 0.0;
            }

            // Wait for playback to complete
            // Timeout: max ~10 minutes (matches MAX_SAMPLES at 48kHz)
            const MAX_WAIT_MS: u32 = 600_000;
            let mut waited_ms: u32 = 0;
            while (hdr.dwFlags & WHDR_DONE) == 0 {
                if waited_ms >= MAX_WAIT_MS {
                    waveOutReset(hwo); // force stop
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
                waited_ms += 10;
            }

            waveOutUnprepareHeader(hwo, &mut hdr, hdr_size);
            waveOutClose(hwo);
            1.0
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (samples, sample_rate);
        0.0
    }
}

/// Play a WAV file asynchronously via PlaySoundW.
/// Returns 1.0 on success, 0.0 on failure.
pub fn audio_play_file_impl(path: &str) -> f32 {
    #[cfg(target_os = "windows")]
    {
        use crate::platform::win32::*;

        let w_path = to_wide(path);
        unsafe {
            let ok = PlaySoundW(
                w_path.as_ptr(),
                std::ptr::null_mut(),
                SND_FILENAME | SND_ASYNC | SND_NODEFAULT,
            );
            if ok != 0 { 1.0 } else { 0.0 }
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = path;
        0.0
    }
}

/// Stop any currently playing audio started by audio_play_file.
pub fn audio_stop_impl() -> f32 {
    #[cfg(target_os = "windows")]
    {
        use crate::platform::win32::*;
        unsafe {
            PlaySoundW(std::ptr::null(), std::ptr::null_mut(), 0);
        }
        1.0
    }
    #[cfg(not(target_os = "windows"))]
    { 0.0 }
}
