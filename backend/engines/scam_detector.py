"""
Scam Content Detection Engine.
Multilingual keyword detection (English / Hindi / Tamil).
Pattern-based scoring with combo multiplier logic.
"""
import re
from typing import Dict, List, Tuple

# ========== KEYWORD DICTIONARIES ==========

SCAM_KEYWORDS = {
    "authority_impersonation": {
        "en": ["police", "government", "federal", "bureau", "IRS", "tax department",
               "reserve bank", "RBI", "customs", "immigration", "social security",
               "ministry", "officer", "inspector", "commissioner", "FBI", "CIA",
               "interpol", "cybercrime", "enforcement"],
        "hi": ["police", "sarkar", "sarkari", "vibhag", "adhikari", "pradhan mantri",
               "mukhyamantri", "tax vibhag", "CBI", "aayog", "nirdeshak"],
        "ta": ["kaavalar", "arasu", "thuravai", "adhikari", "varisulkam thuravai",
               "kaaval thurai", "neethi mandram"],
    },
    "legal_threats": {
        "en": ["arrest", "warrant", "lawsuit", "court", "legal action", "prosecution",
               "prison", "jail", "sentence", "criminal", "felony", "penalty",
               "charges", "violation", "summons", "tribunal", "hearing",
               "non-bailable", "FIR", "complaint filed"],
        "hi": ["giraftaar", "giraftari", "warrant", "adalat", "karvai", "jail",
               "saza", "jurm", "case", "mukadma", "FIR", "shikayat"],
        "ta": ["kaidhhu", "needhimandram", "sattam", "thandanai", "kutrach chaattu",
               "vazhakku", "sirai"],
    },
    "urgency_pressure": {
        "en": ["immediately", "immediate", "urgent", "right now", "within 24 hours",
               "deadline", "expires", "last chance", "final notice", "time is running",
               "hurry", "quickly", "before it's too late", "act now", "don't delay",
               "within the hour", "suspend", "block", "freeze"],
        "hi": ["turant", "abhi", "jaldi", "fauran", "samay khatam", "aakhri mauka",
               "deri mat karo", "band ho jayega", "block ho jayega"],
        "ta": ["udanadi", "ippodhu", "viraivaaga", "kadaisi vaaippu",
               "neram mudiyudhu", "thadai seyyappadum"],
    },
    "payment_requests": {
        "en": ["payment", "transfer", "send money", "bank account", "wire transfer",
               "gift card", "bitcoin", "cryptocurrency", "western union", "money order",
               "pay now", "processing fee", "clearance fee", "refundable deposit",
               "Google Pay", "PayPal", "Venmo", "UPI", "NEFT", "RTGS"],
        "hi": ["bhugtaan", "paisa bhejo", "transfer karo", "bank account",
               "Google Pay", "PhonePe", "Paytm", "UPI", "NEFT"],
        "ta": ["panam", "panam anuppu", "maathru", "vanga kanakku",
               "Google Pay", "PhonePe", "Paytm"],
    },
    "otp_requests": {
        "en": ["OTP", "one-time password", "verification code", "security code",
               "PIN", "CVV", "card number", "password", "login credentials",
               "secret code", "authentication code", "confirm your identity"],
        "hi": ["OTP", "code", "password", "PIN", "gupth code"],
        "ta": ["OTP", "code", "password", "PIN", "ragasiya code"],
    },
    "isolation_instructions": {
        "en": ["don't tell anyone", "keep this confidential", "secret",
               "between us only", "don't hang up", "don't disconnect",
               "stay on the line", "don't call back", "don't contact",
               "don't verify", "do not inform"],
        "hi": ["kisi ko mat batao", "gupth rakho", "phone mat kato",
               "kisi se baat mat karo", "rahasya"],
        "ta": ["yaarukkum sollaatheenga", "ragasiyam", "phone vaikkaatheenga"],
    },
    "emotional_manipulation": {
        "en": ["help your family", "loved ones in danger", "save yourself",
               "life is at risk", "children will suffer", "emergency",
               "accident", "hospital", "kidnapped", "ransom", "threat to life",
               "die", "kill", "bomb", "terror"],
        "hi": ["parivaar khatre mein", "bachcha", "accident", "hospital",
               "jaan ka khatra", "apatkal", "kidnap", "firouti"],
        "ta": ["kudumbam aabathil", "kuzhandhai", "aabathu", "uyirukkuaabathu",
               "kadathhal", "meehppu panam"],
    },
}

# Score weights per category
CATEGORY_WEIGHTS = {
    "authority_impersonation": 15,
    "legal_threats": 18,
    "urgency_pressure": 14,
    "payment_requests": 20,
    "otp_requests": 22,
    "isolation_instructions": 12,
    "emotional_manipulation": 16,
}

CATEGORY_LABELS = {
    "authority_impersonation": "Authority Impersonation",
    "legal_threats": "Legal Threat",
    "urgency_pressure": "Urgency Pressure",
    "payment_requests": "Payment Request",
    "otp_requests": "OTP Request",
    "isolation_instructions": "Isolation Instruction",
    "emotional_manipulation": "Emotional Manipulation",
}


def _match_keywords(text: str, keywords: List[str]) -> List[str]:
    """Find matching keywords in text (case-insensitive)."""
    text_lower = text.lower()
    matched = []
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword.lower()))
        if pattern.search(text_lower):
            matched.append(keyword)
    return matched


def analyze_content(text: str) -> dict:
    """
    Analyze transcript text for scam content indicators.
    
    Returns:
        {
            "score": float (0-100),
            "categories": list[str],
            "matched_keywords": list[str],
            "category_details": dict,
            "combo_multiplier": float,
            "threat_category": str,
            "insight": str,
        }
    """
    if not text or not text.strip():
        return {
            "score": 0,
            "categories": [],
            "matched_keywords": [],
            "category_details": {},
            "combo_multiplier": 1.0,
            "threat_category": "None",
            "insight": "No text content to analyze.",
        }
    
    all_matched = []
    category_scores = {}
    active_categories = []
    category_details = {}
    
    for cat_key, lang_dict in SCAM_KEYWORDS.items():
        cat_matched = []
        for lang, keywords in lang_dict.items():
            matches = _match_keywords(text, keywords)
            cat_matched.extend(matches)
        
        cat_matched = list(set(cat_matched))
        
        if cat_matched:
            weight = CATEGORY_WEIGHTS[cat_key]
            match_intensity = min(len(cat_matched) / 3, 1.0)  # Normalize by 3 keywords max per cat
            cat_score = weight * (0.5 + 0.5 * match_intensity)
            category_scores[cat_key] = cat_score
            active_categories.append(CATEGORY_LABELS[cat_key])
            all_matched.extend(cat_matched)
            category_details[CATEGORY_LABELS[cat_key]] = {
                "score": round(cat_score, 1),
                "keywords": cat_matched,
            }
    
    all_matched = list(set(all_matched))
    
    # Combo multiplier: more categories = higher risk
    num_categories = len(active_categories)
    if num_categories >= 5:
        combo_multiplier = 1.5
    elif num_categories >= 3:
        combo_multiplier = 1.3
    elif num_categories >= 2:
        combo_multiplier = 1.15
    else:
        combo_multiplier = 1.0
    
    # Calculate raw score
    raw_score = sum(category_scores.values())
    final_score = min(100, raw_score * combo_multiplier)
    
    # Determine threat category
    threat_category = "None"
    if final_score >= 60:
        if "otp_requests" in category_scores and "payment_requests" in category_scores:
            threat_category = "Financial Vishing (Voice Phishing)"
        elif "authority_impersonation" in category_scores:
            threat_category = "Government Impersonation Scam"
        elif "emotional_manipulation" in category_scores:
            threat_category = "Emotional Manipulation Scam"
        else:
            threat_category = "General Voice Scam"
    elif final_score >= 30:
        threat_category = "Suspicious Activity"
    
    # Generate insight
    insight_parts = []
    if "urgency_pressure" in category_scores:
        insight_parts.append("High urgency language detected")
    if "otp_requests" in category_scores:
        insight_parts.append("OTP request detected")
    if "payment_requests" in category_scores:
        insight_parts.append("Financial transaction pressure detected")
    if "authority_impersonation" in category_scores:
        insight_parts.append("Authority impersonation patterns identified")
    if "legal_threats" in category_scores:
        insight_parts.append("Legal intimidation tactics used")
    
    insight = ". ".join(insight_parts) + "." if insight_parts else "No significant scam indicators found."
    
    return {
        "score": round(final_score, 1),
        "categories": active_categories,
        "matched_keywords": all_matched,
        "category_details": category_details,
        "combo_multiplier": round(combo_multiplier, 2),
        "threat_category": threat_category,
        "insight": insight,
    }
