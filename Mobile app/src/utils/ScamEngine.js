/**
 * ScamShield Client-Side Analysis Engine
 * Ported from Python scam_detector.py for Mobile MVP
 */

const SCAM_KEYWORDS = {
  authority_impersonation: [
    "police", "government", "federal", "bureau", "IRS", "tax department",
    "reserve bank", "RBI", "customs", "immigration", "social security",
    "ministry", "officer", "inspector", "commissioner", "FBI", "CIA",
    "interpol", "cybercrime", "enforcement", "sarkar", "sarkari", "adhikari",
    "kaavalar", "arasu"
  ],
  legal_threats: [
    "arrest", "warrant", "lawsuit", "court", "legal action", "prosecution",
    "prison", "jail", "sentence", "criminal", "felony", "penalty",
    "charges", "violation", "summons", "tribunal", "hearing",
    "non-bailable", "FIR", "complaint filed", "giraftaar", "adalat", "kaidhhu"
  ],
  urgency_pressure: [
    "immediately", "immediate", "urgent", "right now", "within 24 hours",
    "deadline", "expires", "last chance", "final notice", "time is running",
    "hurry", "quickly", "before it's too late", "act now", "don't delay",
    "within the hour", "suspend", "block", "freeze", "turant", "abhi", "udanadi"
  ],
  payment_requests: [
    "payment", "transfer", "send money", "bank account", "wire transfer",
    "gift card", "bitcoin", "cryptocurrency", "western union", "money order",
    "pay now", "processing fee", "clearance fee", "refundable deposit",
    "Google Pay", "PayPal", "Venmo", "UPI", "NEFT", "RTGS", "bhugtaan", "paisa bhejo", "panam"
  ],
  otp_requests: [
    "OTP", "one-time password", "verification code", "security code",
    "PIN", "CVV", "card number", "password", "login credentials",
    "secret code", "authentication code", "confirm your identity", "gupth code"
  ],
  isolation_instructions: [
    "don't tell anyone", "keep this confidential", "secret",
    "between us only", "don't hang up", "don't disconnect",
    "stay on the line", "don't call back", "don't contact",
    "don't verify", "do not inform", "kisi ko mat batao", "ragasiyam"
  ],
  emotional_manipulation: [
    "help your family", "loved ones in danger", "save yourself",
    "life is at risk", "children will suffer", "emergency",
    "accident", "hospital", "kidnapped", "ransom", "threat to life",
    "die", "kill", "bomb", "terror", "parivaar khatre mein", "aabathu"
  ],
};

const CATEGORY_WEIGHTS = {
  authority_impersonation: 15,
  legal_threats: 18,
  urgency_pressure: 14,
  payment_requests: 20,
  otp_requests: 22,
  isolation_instructions: 12,
  emotional_manipulation: 16,
};

const CATEGORY_LABELS = {
  authority_impersonation: "Authority Impersonation",
  legal_threats: "Legal Threat",
  urgency_pressure: "Urgency Pressure",
  payment_requests: "Payment Request",
  otp_requests: "OTP Request",
  isolation_instructions: "Isolation Instruction",
  emotional_manipulation: "Emotional Manipulation",
};

/**
 * Analyzes text content for scam indicators locally.
 * @param {string} text - The transcribed text to analyze.
 * @returns {object} - Analysis result matching the backend structure.
 */
export const analyzeTextLocally = (text) => {
  if (!text || text.trim() === "") {
    return {
      risk_score: 0,
      threat_category: "None",
      transcript: "",
      verdict: "HUMAN_VOICE",
      voice_score: 0,
    };
  }

  const textLower = text.toLowerCase();
  const matchedCategories = [];
  let rawScore = 0;
  const matchedKeywords = [];

  Object.entries(SCAM_KEYWORDS).forEach(([category, keywords]) => {
    const foundKeywords = keywords.filter(word => 
      textLower.includes(word.toLowerCase())
    );

    if (foundKeywords.length > 0) {
      const weight = CATEGORY_WEIGHTS[category];
      const intensity = Math.min(foundKeywords.length / 3, 1.0);
      const categoryScore = weight * (0.5 + 0.5 * intensity);
      
      rawScore += categoryScore;
      matchedCategories.push(CATEGORY_LABELS[category]);
      matchedKeywords.push(...foundKeywords);
    }
  });

  // Unique keywords
  const uniqueKeywords = [...new Set(matchedKeywords)];

  // Combo multiplier
  let multiplier = 1.0;
  if (matchedCategories.length >= 5) multiplier = 1.5;
  else if (matchedCategories.length >= 3) multiplier = 1.3;
  else if (matchedCategories.length >= 2) multiplier = 1.15;

  const finalScore = Math.min(100, Math.round(rawScore * multiplier));

  // Determine threat category
  let threatCategory = "None";
  if (finalScore >= 60) {
    if (matchedCategories.includes("OTP Request") && matchedCategories.includes("Payment Request")) {
      threatCategory = "Financial Vishing (Voice Phishing)";
    } else if (matchedCategories.includes("Authority Impersonation")) {
      threatCategory = "Government Impersonation Scam";
    } else {
      threatCategory = "General Voice Scam";
    }
  } else if (finalScore >= 30) {
    threatCategory = "Suspicious Activity";
  }

  return {
    risk_score: finalScore,
    threat_category: threatCategory,
    transcript: text,
    verdict: finalScore > 70 ? "AI_VOICE" : "HUMAN_VOICE", // Simulating voice verdict based on content for MVP
    voice_score: finalScore > 70 ? 85 : 12,
    categories: matchedCategories,
    keywords: uniqueKeywords
  };
};
