"""
Risk Fusion Engine — Stable Additive Architecture
==================================================
Core principle:
  • Scam probability is TEXT-ONLY. Voice can only ADD suspicion, never reduce it.
  • Formula: final = clamp(scam_prob + keyword_boost + voice_boost, 0, 1)
  • Negation phrases subtract up to 0.25.
  • Strong intent combo forces final >= 0.65.

Thresholds:
  0.00 – 0.30  → Low
  0.30 – 0.60  → Medium
  0.60+        → High Risk
"""

import re
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger("scamshield.risk_fusion")

# ── Keyword boost table (phrase → weight) ─────────────────────────────────────
# Capped at KEYWORD_BOOST_MAX total. These layer ON TOP of the content score.
KEYWORD_BOOST_MAP: dict[str, float] = {
    # OTP / credential
    "otp": 0.07, "one-time password": 0.07, "verification code": 0.06,
    "security code": 0.06, "pin": 0.04, "cvv": 0.05, "password": 0.04,
    # Payment
    "send money": 0.08, "transfer": 0.06, "bank account": 0.05,
    "upi": 0.05, "gift card": 0.07, "bitcoin": 0.07, "wire transfer": 0.07,
    "google pay": 0.05, "phonepe": 0.05, "paytm": 0.05,
    # Urgency
    "immediately": 0.05, "urgent": 0.05, "right now": 0.05,
    "within 24 hours": 0.06, "last chance": 0.06, "act now": 0.05,
    "suspend": 0.04, "block": 0.04, "freeze": 0.04,
    # Authority / legal
    "arrest": 0.07, "warrant": 0.07, "court": 0.05, "fir": 0.05,
    "police": 0.04, "rbi": 0.04, "irs": 0.04, "government": 0.03,
    # Isolation
    "don't tell anyone": 0.07, "keep this confidential": 0.06,
    "don't hang up": 0.06, "stay on the line": 0.05,
}
KEYWORD_BOOST_MAX = 0.20

# ── Negation phrases ───────────────────────────────────────────────────────────
NEGATION_PHRASES = [
    r"don.?t share.*?otp", r"never give.*?otp", r"never share.*?otp",
    r"do not share.*?otp", r"this is not a scam", r"this is a legitimate",
    r"official call", r"i am not asking for.*?(otp|password|pin)",
    r"never ask.*?(otp|password|card)",
]
NEGATION_DISCOUNT_MAX = 0.25

# ── Strong intent combos → floor at 0.65 ──────────────────────────────────────
INTENT_SET_A = re.compile(
    r"\b(otp|transfer|verify account|verification code|send money|bank account)\b",
    re.IGNORECASE
)
INTENT_SET_B = re.compile(
    r"\b(urgent|immediately|arrest|warrant|police|court|penalty|deadline|"
    r"right now|act now|don.?t delay|hurry)\b",
    re.IGNORECASE
)
STRONG_INTENT_FLOOR = 0.65


# ── Helpers ───────────────────────────────────────────────────────────────────

def _keyword_boost(transcript: str) -> float:
    """Sum boost weights for matched phrases (capped at KEYWORD_BOOST_MAX)."""
    if not transcript:
        return 0.0
    t = transcript.lower()
    total = 0.0
    for phrase, weight in KEYWORD_BOOST_MAP.items():
        if phrase in t:
            total += weight
    return round(min(total, KEYWORD_BOOST_MAX), 4)


def _negation_discount(transcript: str) -> float:
    """Return a discount (0.0–NEGATION_DISCOUNT_MAX) when protective phrases found."""
    if not transcript:
        return 0.0
    t = transcript.lower()
    hits = sum(1 for pat in NEGATION_PHRASES if re.search(pat, t))
    if hits == 0:
        return 0.0
    # Graduated: 1 hit → 0.10, 2 hits → 0.18, 3+ hits → 0.25
    discount = min(0.10 * hits + 0.03 * max(0, hits - 1), NEGATION_DISCOUNT_MAX)
    return round(discount, 4)


def _strong_intent_floor(transcript: str, current: float) -> float:
    """If transcript has both a demand AND an urgency/threat, floor final at 0.65."""
    if not transcript:
        return current
    if INTENT_SET_A.search(transcript) and INTENT_SET_B.search(transcript):
        return max(current, STRONG_INTENT_FLOOR)
    return current


def _level_from_score(s: float) -> str:
    if s < 0.30:
        return "Low"
    if s < 0.60:
        return "Medium"
    return "High"


# ── Main public function ──────────────────────────────────────────────────────

def compute_risk(
    content_score: float,          # 0–100 from scam_detector.analyze_content
    voice_score: float,            # 0–100 synthetic_probability from voice_analyzer
    transcript: str = "",          # raw transcript for negation / strong-intent checks
    voice_verdict: str = "",       # "AI_VOICE" | "HUMAN_VOICE" | "UNCERTAIN"
) -> dict:
    """
    Clean additive risk fusion.

    Rules:
      1. scam_prob derived from content_score alone (0–1).
      2. keyword_boost adds at most +0.20.
      3. voice_boost adds at most +0.15 — ONLY if voice is suspicious.
         Voice NEVER reduces the score.
      4. Negation phrases subtract up to 0.25.
      5. Strong intent combo (demand + urgency/threat) floors score at 0.65.
      6. Final score clamped to [0, 1].
    """
    # ── 1. Normalise inputs ──────────────────────────────────────────────────
    content_score = max(0.0, min(100.0, float(content_score)))
    voice_score   = max(0.0, min(100.0, float(voice_score)))

    # ── 2. Scam probability — text only ─────────────────────────────────────
    scam_prob = content_score / 100.0

    # ── 3. Keyword boost ────────────────────────────────────────────────────
    kw_boost = _keyword_boost(transcript)

    # ── 4. Voice anomaly boost (additive only, never subtracts) ─────────────
    # voice_score is synthetic_probability: 0 = human, 100 = AI
    # Raised ceiling to 0.40 so AI voice alone can reach Medium (0.30+)
    if voice_score >= 50:
        voice_boost = round(((voice_score - 50) / 50) * 0.40, 4)
    else:
        voice_boost = 0.0          # human-sounding voice → no penalty, no benefit

    # If ensemble verdict explicitly says AI, guarantee minimum 0.10 boost
    if voice_verdict == "AI_VOICE" and voice_boost < 0.10:
        voice_boost = 0.10

    # ── 5. Combine ──────────────────────────────────────────────────────────
    raw = scam_prob + kw_boost + voice_boost

    # ── 6. Negation discount ─────────────────────────────────────────────────
    neg = _negation_discount(transcript)
    raw = max(0.0, raw - neg)

    # ── 7. Strong intent floor ───────────────────────────────────────────────
    raw = _strong_intent_floor(transcript, raw)

    # ── 7b. AI Voice floor — ensures AI detection is always visible ──────────
    # If the model is confident it's AI voice, floor at Medium entry (0.35)
    # regardless of transcript content (pure AI voice scam with no keywords)
    if voice_verdict == "AI_VOICE" and voice_score >= 60:
        raw = max(raw, 0.35)

    # ── 8. Clamp ─────────────────────────────────────────────────────────────
    final = round(min(1.0, raw), 4)

    level = _level_from_score(final)

    # Confidence: how well the text and voice signals agree
    # High when both clearly scam (not when voice suppresses text)
    signal_strength = min(1.0, (scam_prob + kw_boost) + voice_boost / 2)
    confidence = round(40.0 + signal_strength * 55.0, 1)
    confidence = min(97.0, max(35.0, confidence))

    # ── Debug log ────────────────────────────────────────────────────────────
    logger.debug(
        "[RiskFusion] transcript_snippet=%r  scam_prob=%.3f  kw_boost=%.3f  "
        "voice_boost=%.3f  neg=%.3f  strong_intent_floor=%s  "
        "final=%.3f  level=%s",
        (transcript[:60] if transcript else ""),
        scam_prob, kw_boost, voice_boost, neg,
        ("YES" if _strong_intent_floor(transcript, 0.0) >= STRONG_INTENT_FLOOR else "no"),
        final, level,
    )
    print(
        f"[RiskFusion] "
        f"scam_prob={scam_prob:.3f}  kw_boost={kw_boost:.3f}  "
        f"voice_boost={voice_boost:.3f}  neg={neg:.3f}  "
        f"final={final:.3f}  level={level}"
    )

    return {
        "final_score": round(final * 100, 1),   # 0–100 for UI compat
        "final_ratio": final,                    # 0–1 for internal use
        "level":       level,
        "confidence":  confidence,
        "breakdown": {
            "scam_prob":    round(scam_prob,    4),
            "keyword_boost": round(kw_boost,   4),
            "voice_boost":   round(voice_boost, 4),
            "neg_discount":  round(neg,          4),
        },
    }


# ── Live mode: rolling stabilizer ────────────────────────────────────────────

class LiveRiskStabilizer:
    """
    Keeps a rolling window of the last N chunk scores.
    Triggers HIGH only if threshold exceeded for `require_consecutive` chunks in a row.
    """

    def __init__(self, window: int = 3, require_consecutive: int = 2):
        self._window = deque(maxlen=window)
        self._consecutive_high = 0
        self._require = require_consecutive
        self._rolling_content  = 0.0
        self._rolling_voice    = 0.0
        self._chunk_count      = 0
        self._alpha            = 0.65   # weight for new chunk vs history

    def update(
        self,
        content_score: float,
        voice_score: float,
        transcript: str = "",
        voice_verdict: str = "",
    ) -> dict:
        """Feed one chunk, get stabilized risk back."""
        self._chunk_count += 1
        α = self._alpha

        # Rolling exponential moving average of raw input signals
        self._rolling_content = self._rolling_content * (1 - α) + content_score * α
        self._rolling_voice   = self._rolling_voice   * (1 - α) + voice_score   * α

        # Fuse with the clean additive formula
        risk = compute_risk(
            self._rolling_content,
            self._rolling_voice,
            transcript=transcript,
            voice_verdict=voice_verdict,
        )

        chunk_final = risk["final_ratio"]
        self._window.append(chunk_final)

        # Rolling average of window
        avg = sum(self._window) / len(self._window)

        # Consecutive HIGH counter — prevents single noisy chunk from alarming
        if chunk_final >= 0.60:
            self._consecutive_high += 1
        else:
            self._consecutive_high = 0

        stable_level = risk["level"]
        if stable_level == "High" and self._consecutive_high < self._require:
            stable_level = "Medium"   # Suppress until confirmed

        stable_score = round(avg * 100, 1)

        print(
            f"[LiveStabilizer] chunk={self._chunk_count}  "
            f"chunk_score={chunk_final:.3f}  "
            f"rolling_avg={avg:.3f}  "
            f"consec_high={self._consecutive_high}  "
            f"stable_level={stable_level}"
        )

        return {
            "final_score":   stable_score,
            "final_ratio":   round(avg, 4),
            "level":         stable_level,
            "confidence":    risk["confidence"],
            "chunk_number":  self._chunk_count,
            "breakdown":     risk["breakdown"],
        }
