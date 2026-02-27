/**
 * ScamShield Audio Analysis Engine v5
 *
 * VOICE AI CLASSIFIER - KNN Implementation
 * Using ALL 200 files from the AiVoiceDataset for direct comparison.
 */

// Import the KNN dataset generated from the local AiVoiceDataset
import knnData from './knn_dataset.json';

/**
 * Returns synthetic_probability (0-100) using KNN (K-Nearest Neighbors)
 * k=5 is used for voting. Distances are Euclidean in z-score normalized space.
 */
function classifyVoice(features) {
  if (!knnData || !knnData.samples || knnData.samples.length === 0) {
    console.error("KNN Dataset not loaded");
    return 50;
  }

  // 1. Construct the query vector in the same order as in knnData.features
  const queryVec = knnData.features.map(fKey => {
    const val = features[fKey];
    return (val !== undefined && val !== null && isFinite(val)) ? val : 0;
  });

  // 2. Normalize the query vector using global mean and std from the dataset
  const normalizedQuery = queryVec.map((val, i) => {
    const mean = knnData.feat_mean[i];
    const std = knnData.feat_std[i];
    return (val - mean) / (std || 1e-6);
  });

  // 3. Compute Euclidean distances to all 200 samples
  const neighbors = knnData.samples.map(sample => {
    // Normalize dataset sample vector
    const normalizedSample = sample.v.map((val, i) => {
      const mean = knnData.feat_mean[i];
      const std = knnData.feat_std[i];
      return (val - mean) / (std || 1e-6);
    });

    // Compute distance
    let distSq = 0;
    for (let i = 0; i < normalizedQuery.length; i++) {
      const diff = normalizedQuery[i] - normalizedSample[i];
      distSq += diff * diff;
    }
    return { label: sample.label, dist: Math.sqrt(distSq) };
  });

  // 4. Sort by distance and take top K=7
  neighbors.sort((a, b) => a.dist - b.dist);
  const k = 7;
  const topK = neighbors.slice(0, k);

  // 5. Compute probability based on weighted votes
  // Distances closer to zero have higher weight
  let aiWeight = 0;
  let humanWeight = 0;
  let totalWeight = 0;

  topK.forEach(n => {
    const w = 1 / (n.dist + 0.1); // Avoid division by zero
    if (n.label === 'ai') aiWeight += w;
    else humanWeight += w;
    totalWeight += w;
  });

  const prob = (aiWeight / totalWeight) * 100;
  
  // Refine: if the very closest neighbor is VERY close, push the probability
  const closest = topK[0];
  let finalProb = prob;
  if (closest.dist < 0.5) {
    if (closest.label === 'ai') finalProb = Math.max(finalProb, 90);
    else finalProb = Math.min(finalProb, 10);
  }

  return Math.round(Math.min(99, Math.max(1, finalProb)) * 10) / 10;
}


// ─── Scam Keywords (EN / HI / TA) ───────────────────────────────────────────
const SCAM_KEYWORDS = {
  authority_impersonation: {
    label: 'Authority Impersonation', weight: 15,
    keywords: [
      'police','government','federal','bureau','irs','tax department',
      'reserve bank','rbi','customs','immigration','social security',
      'ministry','officer','inspector','commissioner','fbi','cia',
      'cybercrime','enforcement','department','authority','official',
      'calling from','headquarters','central','national',
      'sarkar','sarkari','vibhag','adhikari','pradhan mantri',
      'kaavalar','arasu','thuravai','varisulkam',
    ],
  },
  legal_threats: {
    label: 'Legal Threat', weight: 18,
    keywords: [
      'arrest','warrant','lawsuit','court','legal action','prosecution',
      'prison','jail','sentence','criminal','felony','penalty',
      'charges','violation','summons','tribunal','hearing','case filed',
      'non-bailable','fir','complaint','punish','prosecute','guilty',
      'giraftaar','giraftari','adalat','saza','mukadma',
      'kaidhhu','needhimandram','thandanai','vazhakku','sirai',
    ],
  },
  urgency_pressure: {
    label: 'Urgency Pressure', weight: 14,
    keywords: [
      'immediately','immediate','urgent','right now','within 24 hours',
      'deadline','expire','last chance','final notice','time is running',
      'hurry','quickly','too late','act now','delay','now or never',
      'within the hour','suspend','block','freeze','cancel','today only',
      'running out','must act','limited time','right away',
      'turant','abhi','jaldi','fauran','samay khatam',
      'udanadi','ippodhu','viraivaaga','kadaisi vaaippu',
    ],
  },
  payment_requests: {
    label: 'Payment Request', weight: 20,
    keywords: [
      'payment','transfer','send money','bank account','wire',
      'gift card','bitcoin','cryptocurrency','western union','money order',
      'pay now','processing fee','clearance fee','refundable deposit',
      'google pay','paypal','venmo','upi','neft','rtgs','paytm',
      'phonepe','amount','rupees','dollars','transaction','deposit',
      'bhugtaan','paisa','transfer karo',
      'panam','panam anuppu','maathru',
    ],
  },
  otp_requests: {
    label: 'OTP Request', weight: 22,
    keywords: [
      'otp','one time password','one-time','verification code','security code',
      'pin number','pin code','cvv','card number','password',
      'login credentials','secret code','authentication','confirm identity',
      'verify your','share the code','tell me the code','read the number',
      'gupth code','ragasiya code','code sollu',
    ],
  },
  isolation_instructions: {
    label: 'Isolation Instruction', weight: 12,
    keywords: [
      "don't tell",'do not tell','tell anyone','confidential','secret',
      'between us',"don't hang up",'do not hang',"don't disconnect",
      'stay on the line',"don't call back",'do not contact',
      "don't verify",'do not inform','keep quiet','tell nobody',
      'kisi ko mat batao','gupth rakho','phone mat kato',
      'yaarukkum sollaatheenga','ragasiyam',
    ],
  },
  emotional_manipulation: {
    label: 'Emotional Manipulation', weight: 16,
    keywords: [
      'family','loved ones','danger','save yourself','life at risk',
      'children','suffer','emergency','accident','hospital',
      'kidnapped','ransom','threat','die','kill','bomb','terror',
      'help me','please help','scared','afraid','trouble',
      'parivaar','khatre','jaan ka khatra','apatkal','firouti',
      'kudumbam','aabathu','uyirukkuaabathu',
    ],
  },
};

export const HIGHLIGHT_KEYWORDS = Object.values(SCAM_KEYWORDS)
  .flatMap(c => c.keywords).filter(k => k.length > 3);


// ─── Content Analysis ───────────────────────────────────────────────────────
export function analyzeContent(text) {
  if (!text || !text.trim() || text.startsWith('[')) {
    return {
      score: 0, categories: [], matchedKeywords: [], categoryDetails: {},
      comboMultiplier: 1.0, threatCategory: 'None',
      insight: 'No speech content detected yet.',
    };
  }
  const textLower = text.toLowerCase();
  const allMatched = [], activeCategories = [];
  const categoryDetails = {};
  let totalScore = 0;

  for (const [, cat] of Object.entries(SCAM_KEYWORDS)) {
    const matched = cat.keywords.filter(kw => textLower.includes(kw));
    if (matched.length > 0) {
      const intensity  = Math.min(matched.length / 3, 1.0);
      const catScore   = cat.weight * (0.5 + 0.5 * intensity);
      totalScore      += catScore;
      activeCategories.push(cat.label);
      allMatched.push(...matched);
      categoryDetails[cat.label] = {
        score: Math.round(catScore * 10) / 10,
        keywords: [...new Set(matched)],
      };
    }
  }

  const uniqueMatched = [...new Set(allMatched)];
  const numCats  = activeCategories.length;
  const combo    = numCats >= 5 ? 1.5 : numCats >= 3 ? 1.3 : numCats >= 2 ? 1.15 : 1.0;
  const finalScore = Math.min(100, totalScore * combo);

  let threatCategory = 'None';
  if (finalScore >= 60) {
    if (categoryDetails['OTP Request'] && categoryDetails['Payment Request'])
      threatCategory = 'Financial Vishing (Voice Phishing)';
    else if (categoryDetails['Authority Impersonation'])
      threatCategory = 'Government Impersonation Scam';
    else if (categoryDetails['Emotional Manipulation'])
      threatCategory = 'Emotional Manipulation Scam';
    else
      threatCategory = 'General Voice Scam';
  } else if (finalScore >= 30) {
    threatCategory = 'Suspicious Activity';
  }

  const parts = [];
  if (categoryDetails['Urgency Pressure'])        parts.push('High urgency language detected');
  if (categoryDetails['OTP Request'])             parts.push('OTP/credential request detected');
  if (categoryDetails['Payment Request'])          parts.push('Financial transaction pressure detected');
  if (categoryDetails['Authority Impersonation']) parts.push('Authority impersonation patterns identified');
  if (categoryDetails['Legal Threat'])            parts.push('Legal intimidation tactics used');
  if (categoryDetails['Emotional Manipulation'])  parts.push('Emotional manipulation detected');
  if (categoryDetails['Isolation Instruction'])   parts.push('Isolation instructions detected');

  return {
    score: Math.round(finalScore * 10) / 10,
    categories: activeCategories,
    matchedKeywords: uniqueMatched,
    categoryDetails,
    comboMultiplier: combo,
    threatCategory,
    insight: parts.length
      ? parts.join('. ') + '.'
      : 'No significant scam indicators found.',
  };
}


// ─── Live Voice Analyzer ────────────────────────────────────────────────────
/** Real-time feature extraction using AnalyserNode connected to mic stream */
export class LiveVoiceAnalyzer {
  constructor(stream) {
    this.audioCtx   = new (window.AudioContext || window.webkitAudioContext)();
    this.source     = this.audioCtx.createMediaStreamSource(stream);
    this.analyser   = this.audioCtx.createAnalyser();
    this.analyser.fftSize                = 2048;
    this.analyser.smoothingTimeConstant  = 0.2;
    this.source.connect(this.analyser);

    this.bufLen   = this.analyser.frequencyBinCount;
    this.freqData = new Float32Array(this.bufLen);
    this.timeData = new Float32Array(this.analyser.fftSize);

    // Raw feature accumulators
    this.pitchValues    = [];
    this.energyValues   = [];
    this.centroidValues = [];
    this.rolloffValues  = [];
    this.flatnessValues = [];
    this.zcrValues      = [];
    this.mfccDeltaAcc   = [];   // flattened delta frame stds
    this.onsetIntervals = [];
    this._lastOnsetTime = null;
    this.silentFrames   = 0;
    this.totalFrames    = 0;

    this._interval = setInterval(() => this._sample(), 100); // 10 Hz
  }

  _sample() {
    this.analyser.getFloatFrequencyData(this.freqData);
    this.analyser.getFloatTimeDomainData(this.timeData);
    this.totalFrames++;

    // RMS energy
    const rms = Math.sqrt(
      this.timeData.reduce((s, v) => s + v * v, 0) / this.timeData.length
    );
    if (rms < 0.01) { this.silentFrames++; return; }
    this.energyValues.push(rms);

    // Onset detection
    if (this.energyValues.length >= 2) {
      const prev = this.energyValues[this.energyValues.length - 2];
      if (rms > prev * 2 && rms > 0.03) {
        const now = performance.now();
        if (this._lastOnsetTime !== null)
          this.onsetIntervals.push(now - this._lastOnsetTime);
        this._lastOnsetTime = now;
      }
    }

    // Pitch via autocorrelation
    const pitch = this._acfPitch();
    if (pitch > 75 && pitch < 500) this.pitchValues.push(pitch);

    // Spectral features from frequency data (in linear magnitude)
    let wCent = 0, wRoll = 0, totalMag = 0, totalMag2 = 0;
    let sqrSum = 0, absSumSq = 0;     // for spectral flatness
    for (let i = 0; i < this.freqData.length; i++) {
      const mag = Math.max(0, Math.pow(10, this.freqData[i] / 20));
      wCent    += i * mag;
      totalMag += mag;
      totalMag2+= mag * mag;
      // flatness components
      sqrSum   += Math.log(mag + 1e-10);
      absSumSq += mag;
    }
    if (totalMag > 0.001) {
      const centroid = wCent / totalMag;
      this.centroidValues.push(centroid);

      // Spectral rolloff (freq bin where 85% energy cumulates)
      let cumul = 0, rollBin = 0;
      const thresh = 0.85 * totalMag;
      for (let i = 0; i < this.freqData.length; i++) {
        cumul += Math.max(0, Math.pow(10, this.freqData[i] / 20));
        if (cumul >= thresh) { rollBin = i; break; }
      }
      this.rolloffValues.push(rollBin);

      // Spectral flatness = geometric mean / arithmetic mean
      const geoMean = Math.exp(sqrSum / this.freqData.length);
      const arithMean= absSumSq / this.freqData.length;
      if (arithMean > 0) this.flatnessValues.push(geoMean / arithMean);
    }

    // ZCR
    let zcr = 0;
    for (let i = 1; i < this.timeData.length; i++)
      if ((this.timeData[i] >= 0) !== (this.timeData[i-1] >= 0)) zcr++;
    this.zcrValues.push(zcr / this.timeData.length);
  }

  _acfPitch() {
    const data = this.timeData;
    const sr   = this.audioCtx.sampleRate;
    const minL = Math.floor(sr / 500);
    const maxL = Math.floor(sr / 75);
    let best = minL, bestC = -Infinity;
    for (let lag = minL; lag <= Math.min(maxL, data.length / 2); lag++) {
      let c = 0, n1 = 0, n2 = 0;
      for (let j = 0; j < data.length - lag; j++) {
        c  += data[j] * data[j + lag];
        n1 += data[j] * data[j];
        n2 += data[j + lag] * data[j + lag];
      }
      const n = Math.sqrt(n1 * n2);
      if (n > 0) c /= n;
      if (c > bestC) { bestC = c; best = lag; }
    }
    return bestC > 0.3 ? sr / best : 0;
  }

  getAnalysis() {
    const MIN = 15; // Minimum samples needed for reliable analysis
    if (this.pitchValues.length < MIN || this.energyValues.length < MIN) {
      return {
        synthetic_probability: 0,
        pitch_variance: 50, mfcc_stability: 50,
        spectral_smoothness: 50, naturalness_score: 50,
        classification: 'Gathering data...', ready: false,
      };
    }

    const pitchCV    = computeCV(this.pitchValues);
    const energyCV   = computeCV(this.energyValues);
    const spectralCV = computeCV(this.centroidValues);
    const rolloffCV  = computeCV(this.rolloffValues);

    // flatness_mean and flatness_cv — key discriminators from dataset
    const flatnessMean = mean(this.flatnessValues);
    const flatnessCV   = computeCV(this.flatnessValues);

    const zcrCV            = computeCV(this.zcrValues);
    const silenceRatio     = this.totalFrames > 0 ? this.silentFrames / this.totalFrames : 0.2;

    // Rhythm regularity from onset intervals
    let rhythmRegularity = 0.6;
    if (this.onsetIntervals.length >= 3) {
      const rrcv = computeCV(this.onsetIntervals);
      rhythmRegularity = Math.max(0, Math.min(1, 1 - rrcv));
    }

    // MFCC delta std proxy: variance of energy changes between frames
    const energyDiffs = [];
    for (let i = 1; i < this.energyValues.length; i++)
      energyDiffs.push(Math.abs(this.energyValues[i] - this.energyValues[i-1]));
    // Scale to match librosa mfcc_delta_std range (~5.5-5.8)
    const mfccDeltaStd = energyDiffs.length > 2
      ? 5.0 + computeCV(energyDiffs) * 3
      : 5.65;

    const features = {
      pitch_cv: pitchCV,
      energy_cv: energyCV,
      spectral_cv: spectralCV,
      rolloff_cv: rolloffCV,
      flatness_mean: flatnessMean,
      flatness_cv: flatnessCV,
      zcr_cv: zcrCV,
      mfcc_delta_std: mfccDeltaStd,
      silence_ratio: silenceRatio,
      rhythm_regularity: rhythmRegularity,
    };

    const syntheticProb = classifyVoice(features);

    // Display values
    const pitchVarianceDisplay = Math.min(100, pitchCV * 400);
    const spectralDisplay      = Math.min(100, Math.max(0, 100 - spectralCV * 150));
    const mfccDisplay          = Math.min(100, Math.max(0, (mfccDeltaStd - 5.0) * 250));

    const classification =
      syntheticProb >= 70 ? 'AI-Generated Voice (High Confidence)' :
      syntheticProb >= 55 ? 'Possibly AI-Generated' :
      syntheticProb >= 35 ? 'Likely Human Voice' :
      'Natural Human Voice';

    return {
      synthetic_probability: syntheticProb,
      pitch_variance:     Math.round(pitchVarianceDisplay * 10) / 10,
      mfcc_stability:     Math.round(mfccDisplay * 10) / 10,
      spectral_smoothness:Math.round(spectralDisplay * 10) / 10,
      naturalness_score:  Math.round((100 - syntheticProb) * 10) / 10,
      classification,
      ready: true,
      _raw: features,
    };
  }

  destroy() {
    if (this._interval) clearInterval(this._interval);
    try { this.source.disconnect(); } catch (_) {}
    try { this.audioCtx.close(); } catch (_) {}
  }
}


// ─── Offline Buffer Analysis (for uploaded files) ────────────────────────────
// Multi-signal AI voice detection.
// Six independent acoustic signals are fused; KNN is just one of them.
// This ensures AI audio is detected even outside the KNN training distribution.
export function analyzeAudioBuffer(audioBuffer) {
  const data = audioBuffer.getChannelData(0);
  const sr   = audioBuffer.sampleRate;

  if (data.length < sr * 0.5) {
    return _voiceResult(0, 'Too short — cannot analyse', {});
  }

  // ── Frame setup ──────────────────────────────────────────────────────────
  const frameSize = Math.floor(sr * 0.025); // 25 ms
  const hopSize   = Math.floor(frameSize / 2);

  const pitchValues   = [];
  const pitchPeaks    = []; // ACF peak strength per frame
  const energyValues  = [];
  const centroidValues= [];
  const flatnessValues= [];
  const rolloffValues = [];
  const zcrValues     = [];
  let silentFrames = 0, totalFrames = 0;

  for (let i = 0; i + frameSize < data.length; i += hopSize) {
    totalFrames++;
    const frame = data.slice(i, i + frameSize);
    const rms   = Math.sqrt(frame.reduce((a, v) => a + v * v, 0) / frame.length);
    if (rms < 0.008) { silentFrames++; continue; }
    energyValues.push(rms);

    // Pitch + ACF peak strength
    const { hz, peak } = acfPitchFull(frame, sr);
    if (hz > 75 && hz < 500) {
      pitchValues.push(hz);
      pitchPeaks.push(peak); // near 1.0 = very periodic (AI trait)
    }

    // Spectral features (time-domain approximations)
    let wc = 0, te = 0, logSum = 0, absSum = 0;
    const N = frame.length;
    for (let k = 0; k < N; k++) {
      const e = frame[k] * frame[k];
      wc += k * e; te += e;
      logSum += Math.log(e + 1e-10);
      absSum += e;
    }
    if (te > 1e-8) {
      centroidValues.push(wc / te);
      let cumul = 0;
      const thresh = 0.85 * te;
      let roll = 0;
      for (let k = 0; k < N; k++) {
        cumul += frame[k] * frame[k];
        if (cumul >= thresh) { roll = k; break; }
      }
      rolloffValues.push(roll);
      const geo   = Math.exp(logSum / N);
      const arith = absSum / N;
      if (arith > 0) flatnessValues.push(geo / arith);
    }

    let zcr = 0;
    for (let k = 1; k < frame.length; k++)
      if ((frame[k] >= 0) !== (frame[k-1] >= 0)) zcr++;
    zcrValues.push(zcr / frame.length);

    if (pitchValues.length >= 600) break;
  }

  if (pitchValues.length < 5 || energyValues.length < 5) {
    return _voiceResult(0, 'Insufficient voice data', {});
  }

  // ── Signal 1: Pitch CV (coefficient of variation) ──────────────────────
  // Human: pitch_cv > 0.12 (speech melody varies naturally)
  // AI TTS: pitch_cv often < 0.06 (unnaturally flat / controlled)
  const pitchCV   = computeCV(pitchValues);
  const pitchMean = mean(pitchValues);
  const pitchStd  = Math.sqrt(pitchValues.reduce((a,v) => a + (v-pitchMean)**2, 0) / pitchValues.length);
  // Score: low CV → high synthetic probability
  const pitchCVScore = clamp01(1 - pitchCV / 0.20);

  // ── Signal 2: Pitch Jitter (cycle-to-cycle F0 variation) ───────────────
  // Humans: 0.5–1.5 % jitter.  AI/TTS: < 0.1 % (near-perfect periodicity)
  const jitterVals = [];
  for (let i = 1; i < pitchValues.length; i++)
    jitterVals.push(Math.abs(pitchValues[i] - pitchValues[i-1]) / (pitchValues[i-1] + 1e-6));
  const jitter = mean(jitterVals);   // relative jitter
  // Score: very low jitter → high AI probability
  const jitterScore = clamp01(1 - jitter / 0.03);

  // ── Signal 3: ACF Peak Strength (periodicity) ──────────────────────────
  // Humans: ACF peak typically 0.3–0.65.  AI: often > 0.75 (too periodic)
  const acfMean  = mean(pitchPeaks);
  const acfScore = clamp01((acfMean - 0.50) / 0.35); // > 0.85 → score ≈ 1

  // ── Signal 4: Shimmer (amplitude variation between pitch cycles) ────────
  // Humans: ~3-8 % shimmer.  AI: < 1 % (mechanically consistent amplitude)
  const shimmerVals = [];
  for (let i = 1; i < energyValues.length; i++)
    shimmerVals.push(Math.abs(energyValues[i] - energyValues[i-1]) / (energyValues[i-1] + 1e-6));
  const shimmer = mean(shimmerVals);
  // Score: very low shimmer → high AI probability
  const shimmerScore = clamp01(1 - shimmer / 0.12);

  // ── Signal 5: Spectral Flatness Stability ─────────────────────────────
  // AI voices have extremely consistent spectral shape across frames
  // Low flatness_cv → more AI-like
  const flatnessCV  = flatnessValues.length >= 3 ? computeCV(flatnessValues) : 5.0;
  const flatScore   = clamp01(1 - flatnessCV / 4.0);

  // ── Signal 6: KNN-based classifier (existing training data) ───────────
  const energyDiffs = [];
  for (let i = 1; i < energyValues.length; i++)
    energyDiffs.push(Math.abs(energyValues[i] - energyValues[i-1]));
  const mfccDeltaStd  = energyDiffs.length > 2 ? 5.0 + computeCV(energyDiffs) * 3 : 5.65;
  const rhythmCV      = computeCV(centroidValues);
  const onsetIntervals = _onsetIntervals(data, sr);
  const rhythmReg = onsetIntervals.length >= 3
    ? Math.max(0, Math.min(1, 1 - computeCV(onsetIntervals)))
    : 0.6;

  const knnFeatures = {
    pitch_cv:          pitchCV,
    energy_cv:         computeCV(energyValues),
    spectral_cv:       centroidValues.length >= 3 ? rhythmCV : 0.69,
    rolloff_cv:        rolloffValues.length >= 3  ? computeCV(rolloffValues) : 0.65,
    flatness_mean:     flatnessValues.length >= 3 ? mean(flatnessValues) : 0.0019,
    flatness_cv:       flatnessCV,
    zcr_cv:            zcrValues.length >= 3 ? computeCV(zcrValues) : 1.12,
    mfcc_delta_std:    mfccDeltaStd,
    silence_ratio:     totalFrames > 0 ? silentFrames / totalFrames : 0.23,
    rhythm_regularity: rhythmReg,
  };
  const knnProb = classifyVoice(knnFeatures); // 0-100

  // ── Fusion: weighted blend of 6 signals ───────────────────────────────
  // Weights are chosen so that each heuristic signal contributes even
  // when KNN has no matching training samples.
  //
  //  Signal          weight  rationale
  //  pitch_cv_score  0.20    primary TTS discriminator
  //  jitter_score    0.20    best single-feature discriminator
  //  acf_score       0.18    periodicity (AI voices are clock-perfect)
  //  shimmer_score   0.15    amplitude naturalness
  //  flat_score      0.12    spectral consistency
  //  knn_score       0.15    learned dataset patterns
  const heuristicProb = (
    pitchCVScore  * 0.20 +
    jitterScore   * 0.20 +
    acfScore      * 0.18 +
    shimmerScore  * 0.15 +
    flatScore     * 0.12 +
    (knnProb / 100) * 0.15
  ) * 100;

  // Soft-max blend: if both KNN and heuristic agree strongly, amplify
  const agreement   = Math.abs(heuristicProb - knnProb) < 20 ? 1.05 : 1.0;
  const syntheticProb = Math.round(clamp(heuristicProb * agreement, 1, 99) * 10) / 10;

  // ── Display metrics ───────────────────────────────────────────────────
  const pitchVarianceDisplay = Math.min(100, pitchCV * 350);
  const jitterDisplay        = Math.min(100, (1 - jitterScore) * 100);
  const spectralDisplay      = Math.min(100, Math.max(0, 100 - rhythmCV * 150));
  const mfccDisplay          = Math.min(100, Math.max(0, (mfccDeltaStd - 5.0) * 250));

  const classification =
    syntheticProb >= 75 ? 'AI-Generated Voice (High Confidence)' :
    syntheticProb >= 55 ? 'Likely AI-Generated / Voice-Converted' :
    syntheticProb >= 38 ? 'Possibly AI-Generated — Borderline' :
    syntheticProb >= 20 ? 'Likely Human Voice' :
    'Natural Human Voice';

  const raw = {
    pitch_cv: pitchCV, pitch_std: pitchStd, pitch_mean: pitchMean,
    jitter, shimmer, acf_mean: acfMean,
    flatness_cv: flatnessCV, knn_prob: knnProb,
    scores: { pitchCVScore, jitterScore, acfScore, shimmerScore, flatScore },
  };

  return _voiceResult(syntheticProb, classification, {
    pitch_variance:      Math.round(pitchVarianceDisplay * 10) / 10,
    mfcc_stability:      Math.round(mfccDisplay * 10) / 10,
    spectral_smoothness: Math.round(spectralDisplay * 10) / 10,
    jitter_score:        Math.round(jitterDisplay * 10) / 10,
    _raw: raw,
  });
}

/** Build the result object for analyzeAudioBuffer. */
function _voiceResult(syntheticProb, classification, extra) {
  return {
    synthetic_probability: syntheticProb,
    naturalness_score:     Math.round((100 - syntheticProb) * 10) / 10,
    classification,
    pitch_variance:      extra.pitch_variance      ?? 50,
    mfcc_stability:      extra.mfcc_stability      ?? 50,
    spectral_smoothness: extra.spectral_smoothness ?? 50,
    jitter_score:        extra.jitter_score        ?? 50,
    _raw:                extra._raw               ?? {},
  };
}

/** Compute onset intervals (for rhythm regularity). */
function _onsetIntervals(data, sr) {
  const stepSize = 1024;
  const intervals = [];
  let prevE = 0, lastOnset = -1;
  for (let i = 0; i + stepSize < data.length; i += stepSize) {
    let rms = 0;
    for (let k = i; k < i + stepSize; k++) rms += data[k] * data[k];
    rms = Math.sqrt(rms / stepSize);
    if (rms > prevE * 2 && rms > 0.03) {
      const fi = Math.floor(i / stepSize);
      if (lastOnset >= 0) intervals.push((fi - lastOnset) * stepSize / sr);
      lastOnset = fi;
    }
    prevE = rms;
  }
  return intervals;
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/** ACF pitch estimator — returns { hz, peak } where peak is the ACF strength (0-1). */
function acfPitchFull(frame, sr) {
  const minL = Math.floor(sr / 500);
  const maxL = Math.floor(sr / 75);
  let best = minL, bestC = -1;
  for (let lag = minL; lag <= Math.min(maxL, frame.length - 1); lag++) {
    let c = 0, n1 = 0, n2 = 0;
    for (let j = 0; j < frame.length - lag; j++) {
      c  += frame[j] * frame[j + lag];
      n1 += frame[j] * frame[j];
      n2 += frame[j + lag] * frame[j + lag];
    }
    const n = Math.sqrt(n1 * n2);
    if (n > 0) c /= n;
    if (c > bestC) { bestC = c; best = lag; }
  }
  if (bestC > 0.25) return { hz: sr / best, peak: bestC };
  return { hz: 0, peak: 0 };
}


// ─── Utilities ───────────────────────────────────────────────────────────────
function computeCV(values) {
  if (!values || values.length < 2) return 0;
  const m = mean(values);
  if (Math.abs(m) < 1e-10) return 0;
  const s = Math.sqrt(values.reduce((a, v) => a + (v - m) ** 2, 0) / values.length);
  return s / Math.abs(m);
}

function mean(values) {
  if (!values || values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}


// ─── Risk Fusion ─────────────────────────────────────────────────────────────
export function computeRisk(contentScore, voiceScore) {
  const c = Math.max(0, Math.min(100, contentScore));
  const v = Math.max(0, Math.min(100, voiceScore));
  const finalScore = Math.round((c * 0.6 + v * 0.4) * 10) / 10;
  const level = finalScore < 30 ? 'Low' : finalScore < 60 ? 'Medium' : 'High';
  const agreement = 100 - Math.abs(c - v);
  const dataConf  = Math.min(100, (c + v) / 2 + 30);
  const confidence= Math.min(99, Math.max(30, Math.round(agreement * 0.4 + dataConf * 0.6)));
  return { finalScore, level, confidence };
}


// ─── Audio File Decoder ───────────────────────────────────────────────────────
export async function decodeAudioFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx    = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  await audioCtx.close();
  return audioBuffer;
}


// ─── Dataset-Based Text Comparison ────────────────────────────────────────────
const _STOPWORDS = new Set([
  'the','and','you','for','this','that','have','with','from','will',
  'your','are','was','not','but','can','all','been','about','our',
  'they','them','their','what','which','who','would','there','could',
  'we','is','it','be','as','at','by','an','so','do','to','of','in',
  'on','or','me','my','his','her','its','shall','just','also','very',
  'please','yes','okay','well','hello','good','thank','thanks','right',
]);

function _tokenize(text) {
  return text
    .toLowerCase()
    .replace(/\[.*?\]/g, '')        // strip [Greetings], [Name], etc.
    .replace(/^\d+\.\s*/gm, '')     // strip leading numbering "1. "
    .replace(/[^a-z0-9\s]/g, ' ')   // remove punctuation
    .split(/\s+/)
    .filter(w => w.length > 2 && !_STOPWORDS.has(w));
}

function _dice(tokensA, tokensB) {
  const a = new Set(tokensA);
  const b = new Set(tokensB);
  if (a.size === 0 || b.size === 0) return 0;
  let hits = 0;
  for (const t of a) if (b.has(t)) hits++;
  return (2 * hits) / (a.size + b.size);
}

/**
 * Compare a transcript against the loaded scam / non-scam sentence lists.
 * Returns { datasetScore (0-100), topScamMatches, topNonScamMatches, ready }.
 */
export function compareWithDataset(transcript, scamLines, nonScamLines) {
  if (!transcript || transcript.trim().length < 8) {
    return { datasetScore: 0, topScamMatches: [], topNonScamMatches: [], ready: false };
  }
  const tTokens = _tokenize(transcript);
  if (tTokens.length < 3) {
    return { datasetScore: 0, topScamMatches: [], topNonScamMatches: [], ready: false };
  }

  const scoreLine = (line) => ({
    text: line.replace(/^\d+\.\s*/, '').substring(0, 140).trim(),
    score: _dice(tTokens, _tokenize(line)),
  });

  const topScam = (scamLines || [])
    .filter(l => l.trim().length > 15)
    .map(scoreLine)
    .sort((a, b) => b.score - a.score)
    .slice(0, 4);

  const topNon = (nonScamLines || [])
    .filter(l => l.trim().length > 15)
    .map(scoreLine)
    .sort((a, b) => b.score - a.score)
    .slice(0, 4);

  // Use avg of top-2 scores to reduce noise
  const avgScam  = (topScam.slice(0, 2).reduce((s, m) => s + m.score, 0) / 2)  || 0;
  const avgNon   = (topNon.slice(0, 2).reduce((s, m) => s + m.score, 0)  / 2)  || 0;
  const total    = avgScam + avgNon;
  const datasetScore = total > 0.004
    ? Math.round((avgScam / total) * 100)
    : 50; // uncertain if no similarity at all

  return {
    datasetScore,
    topScamMatches:    topScam,
    topNonScamMatches: topNon,
    ready: true,
    scamSim:    +(avgScam * 100).toFixed(1),
    nonScamSim: +(avgNon  * 100).toFixed(1),
  };
}
