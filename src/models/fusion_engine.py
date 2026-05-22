"""
Weighted ensemble fusion engine for multi-modal counterfeit drug detection.
Combines NIR spectroscopy, pill image, and packaging OCR scores
into a final authenticity verdict with human-readable evidence summary.
"""

from dataclasses import dataclass
from typing import Optional, List


MODALITY_WEIGHTS = {"nir": 0.50, "vision": 0.30, "packaging": 0.20}
COUNTERFEIT_THRESHOLD = 0.45
SUSPICIOUS_THRESHOLD = 0.30


@dataclass
class FusionResult:
    final_verdict: str          # AUTHENTIC / SUSPICIOUS / LIKELY_COUNTERFEIT
    overall_score: float        # 0 = definitely counterfeit, 1 = definitely authentic
    confidence: float
    nir_score: Optional[float]
    vision_score: Optional[float]
    packaging_score: Optional[float]
    failed_checks: List[str]
    evidence_summary: str
    recommendation: str
    report_to_nafdac: bool


def fuse_scores(
    nir_authentic_prob: Optional[float] = None,
    vision_authentic_prob: Optional[float] = None,
    packaging_authentic_prob: Optional[float] = None,
    drug_code: str = "UNKNOWN",
    facility_id: str = "UNKNOWN",
) -> FusionResult:
    available = {}
    if nir_authentic_prob is not None:
        available["nir"] = nir_authentic_prob
    if vision_authentic_prob is not None:
        available["vision"] = vision_authentic_prob
    if packaging_authentic_prob is not None:
        available["packaging"] = packaging_authentic_prob

    if not available:
        raise ValueError("At least one modality score is required.")

    total_weight = sum(MODALITY_WEIGHTS[m] for m in available)
    weighted_score = sum(
        MODALITY_WEIGHTS[m] * score / total_weight
        for m, score in available.items()
    )

    failed_checks = []
    if nir_authentic_prob is not None and nir_authentic_prob < 0.60:
        failed_checks.append(f"NIR: API not detected or low concentration (score={nir_authentic_prob:.2f})")
    if vision_authentic_prob is not None and vision_authentic_prob < 0.60:
        failed_checks.append(f"Visual: Pill appearance anomalies detected (score={vision_authentic_prob:.2f})")
    if packaging_authentic_prob is not None and packaging_authentic_prob < 0.60:
        failed_checks.append(f"Packaging: NAFDAC number invalid or format violations (score={packaging_authentic_prob:.2f})")

    if weighted_score >= (1 - SUSPICIOUS_THRESHOLD):
        verdict = "AUTHENTIC"
        reco = "Drug appears genuine. Cleared for dispensing."
        report = False
    elif weighted_score >= (1 - COUNTERFEIT_THRESHOLD):
        verdict = "SUSPICIOUS"
        reco = "Do not dispense. Flag for lab confirmation. Hold batch."
        report = True
    else:
        verdict = "LIKELY_COUNTERFEIT"
        reco = "DO NOT DISPENSE. Quarantine batch. Report to NAFDAC immediately. Preserve sample for lab."
        report = True

    evidence_parts = []
    if failed_checks:
        evidence_parts.append("Failed checks: " + "; ".join(failed_checks))
    else:
        evidence_parts.append("All available checks passed.")
    evidence_parts.append(f"Ensemble score: {weighted_score:.3f} (threshold: {1-SUSPICIOUS_THRESHOLD:.2f})")

    return FusionResult(
        final_verdict=verdict,
        overall_score=round(weighted_score, 4),
        confidence=round(min(abs(weighted_score - 0.5) * 2 + 0.5, 0.99), 3),
        nir_score=nir_authentic_prob,
        vision_score=vision_authentic_prob,
        packaging_score=packaging_authentic_prob,
        failed_checks=failed_checks,
        evidence_summary=" | ".join(evidence_parts),
        recommendation=reco,
        report_to_nafdac=report,
    )
