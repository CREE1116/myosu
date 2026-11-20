let audioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  if (!audioCtx) {
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctx) return null;
    audioCtx = new Ctx();
  }
  // While suspended, currentTime is frozen: anything scheduled now would pile up and
  // burst out together on resume. Drop this sound and just wake the context.
  if (audioCtx.state !== 'running') {
    void audioCtx.resume();
    return null;
  }
  return audioCtx;
}

export function playStoneSound(enabled: boolean): void {
  if (!enabled) return;
  const ctx = getAudioContext();
  if (!ctx) return;

  const now = ctx.currentTime;
  const pitchShift = 0.92 + Math.random() * 0.16;

  // Short crisp wooden "tak" using filtered noise + low-mid body oscillator
  const osc = ctx.createOscillator();
  const oscGain = ctx.createGain();
  osc.type = 'triangle';
  osc.frequency.setValueAtTime(220 * pitchShift, now);
  osc.frequency.exponentialRampToValueAtTime(75 * pitchShift, now + 0.055);
  oscGain.gain.setValueAtTime(0.28, now);
  oscGain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);
  osc.connect(oscGain);
  oscGain.connect(ctx.destination);
  osc.start(now);
  osc.stop(now + 0.065);

  const bufferSize = Math.floor(ctx.sampleRate * 0.025);
  const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) {
    data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.28));
  }

  const noise = ctx.createBufferSource();
  noise.buffer = buffer;

  const filter = ctx.createBiquadFilter();
  filter.type = 'bandpass';
  filter.frequency.setValueAtTime(1400 * pitchShift, now);
  filter.Q.setValueAtTime(1.8, now);

  const noiseGain = ctx.createGain();
  noiseGain.gain.setValueAtTime(0.32, now);
  noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.025);

  noise.connect(filter);
  filter.connect(noiseGain);
  noiseGain.connect(ctx.destination);
  noise.start(now);
}

export function playForbiddenSound(enabled: boolean): void {
  if (!enabled) return;
  const ctx = getAudioContext();
  if (!ctx) return;

  const now = ctx.currentTime;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(190, now);
  osc.frequency.linearRampToValueAtTime(135, now + 0.12);
  gain.gain.setValueAtTime(0.14, now);
  gain.gain.exponentialRampToValueAtTime(0.001, now + 0.13);
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(now);
  osc.stop(now + 0.14);
}

export function playWinSound(enabled: boolean): void {
  if (!enabled) return;
  const ctx = getAudioContext();
  if (!ctx) return;

  const now = ctx.currentTime;
  const notes = [523.25, 659.25, 783.99, 1046.5];
  notes.forEach((freq, idx) => {
    const t = now + idx * 0.075;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.12, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.32);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.34);
  });
}

/** A short synthesized "meow": rising-then-falling pitch through a vowel-like formant sweep (ee→ah→oo). */
export function playMeowSound(enabled: boolean, mood: 'happy' | 'curious' = 'happy'): void {
  if (!enabled) return;
  const ctx = getAudioContext();
  if (!ctx) return;

  const now = ctx.currentTime;
  const dur = mood === 'happy' ? 0.55 : 0.4;
  const base = (mood === 'happy' ? 560 : 640) * (0.95 + Math.random() * 0.1);

  const osc = ctx.createOscillator();
  osc.type = 'sawtooth';
  osc.frequency.setValueAtTime(base, now);
  osc.frequency.linearRampToValueAtTime(base * 1.35, now + dur * 0.25);
  osc.frequency.exponentialRampToValueAtTime(base * 0.75, now + dur);

  // light vibrato
  const lfo = ctx.createOscillator();
  const lfoGain = ctx.createGain();
  lfo.frequency.setValueAtTime(7, now);
  lfoGain.gain.setValueAtTime(base * 0.02, now);
  lfo.connect(lfoGain);
  lfoGain.connect(osc.frequency);

  const formant = ctx.createBiquadFilter();
  formant.type = 'bandpass';
  formant.Q.setValueAtTime(4, now);
  formant.frequency.setValueAtTime(2000, now);
  formant.frequency.linearRampToValueAtTime(1300, now + dur * 0.45);
  formant.frequency.linearRampToValueAtTime(750, now + dur);

  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.32, now + 0.05);
  gain.gain.setValueAtTime(0.32, now + dur * 0.6);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + dur);

  osc.connect(formant);
  formant.connect(gain);
  gain.connect(ctx.destination);
  osc.start(now);
  lfo.start(now);
  osc.stop(now + dur + 0.02);
  lfo.stop(now + dur + 0.02);
}
