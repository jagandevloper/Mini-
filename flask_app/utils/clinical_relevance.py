"""
Clinical Relevance Analysis Module
==================================

Advanced clinical interpretation including:
- Severity assessment
- Treatment urgency
- Anatomical relevance
- Patient management recommendations
- Clinical decision support
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class SeverityLevel(Enum):
    """Clinical severity levels."""
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    CRITICAL = "Critical"


class TreatmentUrgency(Enum):
    """Treatment urgency levels."""
    ROUTINE = "Routine Follow-up"
    URGENT = "Urgent Review"
    EMERGENCY = "Emergency Treatment"


class ClinicalRelevanceAnalyzer:
    """Analyzes clinical relevance of kidney stone detections."""
    
    def __init__(self):
        """Initialize clinical relevance analyzer."""
        self.severity_thresholds = {
            'mild': (0, 30),
            'moderate': (30, 60),
            'severe': (60, 85),
            'critical': (85, 100)
        }
    
    def analyze_clinical_relevance(self, detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """
        Comprehensive clinical relevance analysis.
        
        Args:
            detections: List of detection dictionaries
            image_shape: (height, width) of image
            
        Returns:
            Dictionary with clinical relevance analysis
        """
        if not detections:
            return self._no_detections_analysis()
        
        analysis = {
            'severity_assessment': self._assess_severity(detections, image_shape),
            'treatment_urgency': self._determine_urgency(detections),
            'anatomical_relevance': self._analyze_anatomical_relevance(detections, image_shape),
            'clinical_recommendations': self._generate_recommendations(detections),
            'management_guidance': self._provide_management_guidance(detections),
            'follow_up_plan': self._suggest_follow_up(detections),
            'risk_factors': self._identify_risk_factors(detections, image_shape),
            'clinical_interpretation': self._clinical_interpretation(detections)
        }
        
        return analysis
    
    def _assess_severity(self, detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Assess clinical severity of the findings."""
        num_stones = len(detections)
        avg_confidence = sum([d['confidence'] for d in detections]) / num_stones
        
        # Calculate stone characteristics
        h, w = image_shape
        stone_sizes = []
        stone_positions = []
        
        for det in detections:
            bbox = det['bbox']
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])
            area = width * height
            relative_area = area / (w * h)
            stone_sizes.append(relative_area)
            
            # Center position
            cx = (bbox[0] + bbox[2]) / 2 / w
            cy = (bbox[1] + bbox[3]) / 2 / h
            stone_positions.append((cx, cy))
        
        avg_size = np.mean(stone_sizes)
        total_area = sum(stone_sizes)
        max_individual_size = max(stone_sizes) if stone_sizes else 0
        
        # Calculate severity score
        size_score = min(total_area * 1000, 40)  # Max 40 points
        count_score = min(num_stones * 5, 25)     # Max 25 points
        confidence_score = avg_confidence * 20     # Max 20 points
        size_consistency = min(max_individual_size * 500, 15)  # Max 15 points
        
        total_score = size_score + count_score + confidence_score + size_consistency
        
        # Determine severity level
        if total_score >= 85:
            severity = SeverityLevel.CRITICAL
            interpretation = "Multiple large high-confidence stones detected. Immediate clinical attention required."
        elif total_score >= 60:
            severity = SeverityLevel.SEVERE
            interpretation = "Significant kidney stone burden detected. Urgent clinical review recommended."
        elif total_score >= 30:
            severity = SeverityLevel.MODERATE
            interpretation = "Moderate kidney stone presence. Clinical review recommended with monitoring."
        else:
            severity = SeverityLevel.MILD
            interpretation = "Small stones detected. Routine follow-up recommended."
        
        return {
            'severity_level': severity.value,
            'severity_score': round(total_score, 1),
            'severity_breakdown': {
                'size_contribution': round(size_score, 1),
                'count_contribution': round(count_score, 1),
                'confidence_contribution': round(confidence_score, 1),
                'consistency_contribution': round(size_consistency, 1)
            },
            'interpretation': interpretation,
            'num_stones': num_stones,
            'avg_confidence': round(avg_confidence, 3),
            'total_stone_area_relative': round(total_area * 100, 2),
            'largest_stone_relative_size': round(max_individual_size * 100, 2)
        }
    
    def _determine_urgency(self, detections: List[Dict]) -> Dict:
        """Determine treatment urgency."""
        num_stones = len(detections)
        confidences = [d['confidence'] for d in detections]
        avg_confidence = np.mean(confidences)
        high_conf_count = sum([1 for c in confidences if c > 0.7])
        
        # Urgency scoring
        urgency_score = 0
        
        # More stones = higher urgency
        if num_stones >= 5:
            urgency_score += 40
        elif num_stones >= 3:
            urgency_score += 30
        elif num_stones >= 2:
            urgency_score += 20
        else:
            urgency_score += 10
        
        # Higher confidence = higher urgency (more certain pathology)
        urgency_score += avg_confidence * 30
        
        # Large proportion of high-confidence detections
        if high_conf_count / num_stones > 0.7:
            urgency_score += 20
        
        # Determine urgency level
        if urgency_score >= 70:
            urgency = TreatmentUrgency.EMERGENCY
            time_frame = "Within 24-48 hours"
        elif urgency_score >= 40:
            urgency = TreatmentUrgency.URGENT
            time_frame = "Within 1 week"
        else:
            urgency = TreatmentUrgency.ROUTINE
            time_frame = "Within 2-4 weeks"
        
        return {
            'urgency_level': urgency.value,
            'urgency_score': round(urgency_score, 1),
            'recommended_time_frame': time_frame,
            'factors': {
                'stone_count': num_stones,
                'confidence_level': avg_confidence,
                'high_confidence_stones': high_conf_count
            }
        }
    
    def _analyze_anatomical_relevance(self, detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Analyze anatomical relevance of detections."""
        h, w = image_shape
        
        anatomical_regions = {
            'upper_abdomen': [],
            'mid_abdomen': [],
            'lower_abdomen': [],
            'left_kidney': [],
            'right_kidney': [],
            'bladder_region': []
        }
        
        for det in detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Determine anatomical region
            x_norm = cx / w
            y_norm = cy / h
            
            regions = []
            
            # Vertical regions
            if y_norm < 0.3:
                regions.append('Upper Abdomen')
                anatomical_regions['upper_abdomen'].append(det)
            elif y_norm < 0.7:
                regions.append('Mid Abdomen')
                anatomical_regions['mid_abdomen'].append(det)
            else:
                regions.append('Lower Abdomen/Bladder')
                anatomical_regions['lower_abdomen'].append(det)
            
            # Lateral regions
            if x_norm < 0.35:
                regions.append('Left Kidney')
                anatomical_regions['left_kidney'].append(det)
            elif x_norm > 0.65:
                regions.append('Right Kidney')
                anatomical_regions['right_kidney'].append(det)
            elif y_norm > 0.7:
                regions.append('Bladder Region')
                anatomical_regions['bladder_region'].append(det)
            
            det['anatomical_regions'] = regions
        
        # Clinical significance of location
        clinical_significance = []
        
        left_kidney_stones = len(anatomical_regions['left_kidney'])
        right_kidney_stones = len(anatomical_regions['right_kidney'])
        bladder_stones = len(anatomical_regions['bladder_region'])
        
        if left_kidney_stones > 0:
            clinical_significance.append({
                'region': 'Left Kidney',
                'count': left_kidney_stones,
                'clinical_importance': 'High - Can affect kidney function',
                'consideration': 'Monitor renal function, assess for obstruction'
            })
        
        if right_kidney_stones > 0:
            clinical_significance.append({
                'region': 'Right Kidney',
                'count': right_kidney_stones,
                'clinical_importance': 'High - Can affect kidney function',
                'consideration': 'Monitor renal function, assess for obstruction'
            })
        
        if bladder_stones > 0:
            clinical_significance.append({
                'region': 'Bladder',
                'count': bladder_stones,
                'clinical_importance': 'Medium - May cause urinary symptoms',
                'consideration': 'Monitor for urinary symptoms, infection risk'
            })
        
        return {
            'detection_distribution': {
                'left_kidney': left_kidney_stones,
                'right_kidney': right_kidney_stones,
                'bladder': bladder_stones,
                'upper_abdomen': len(anatomical_regions['upper_abdomen']),
                'mid_abdomen': len(anatomical_regions['mid_abdomen']),
                'lower_abdomen': len(anatomical_regions['lower_abdomen'])
            },
            'clinical_significance': clinical_significance,
            'primary_concern_location': 'Kidney' if max(left_kidney_stones, right_kidney_stones) > bladder_stones else 'Bladder'
        }
    
    def _generate_recommendations(self, detections: List[Dict]) -> Dict:
        """Generate clinical recommendations."""
        num_stones = len(detections)
        avg_confidence = np.mean([d['confidence'] for d in detections])
        
        recommendations = []
        
        # Based on stone count
        if num_stones >= 5:
            recommendations.append({
                'priority': 'High',
                'action': 'Consider CT scan for detailed characterization',
                'rationale': 'Multiple stones detected, CT provides 3D localization'
            })
        elif num_stones >= 3:
            recommendations.append({
                'priority': 'Medium',
                'action': 'Consider follow-up imaging',
                'rationale': 'Several stones detected, monitor progression'
            })
        
        # Based on confidence
        if avg_confidence >= 0.8:
            recommendations.append({
                'priority': 'High',
                'action': 'High-confidence detection, prioritize evaluation',
                'rationale': 'Strong AI confidence suggests significant findings'
            })
        
        # General recommendations
        recommendations.extend([
            {
                'priority': 'Standard',
                'action': 'Consult urologist or nephrologist',
                'rationale': 'Specialist evaluation for comprehensive management'
            },
            {
                'priority': 'Standard',
                'action': 'Consider laboratory studies (urinalysis, serum creatinine)',
                'rationale': 'Assess renal function and infection risk'
            }
        ])
        
        return {
            'primary_recommendations': recommendations[:3],
            'all_recommendations': recommendations,
            'adherence_level': self._calculate_adherence(detections)
        }
    
    def _calculate_adherence(self, detections: List[Dict]) -> str:
        """Calculate adherence level based on severity."""
        num_stones = len(detections)
        
        if num_stones >= 5:
            return "Immediate - Multiple stones require prompt attention"
        elif num_stones >= 3:
            return "Urgent - Several stones need evaluation within days"
        else:
            return "Standard - Routine follow-up appropriate"
    
    def _provide_management_guidance(self, detections: List[Dict]) -> Dict:
        """Provide clinical management guidance."""
        num_stones = len(detections)
        avg_confidence = np.mean([d['confidence'] for d in detections])
        high_conf_stones = sum([1 for d in detections if d['confidence'] > 0.7])
        
        # Management approach
        if num_stones >= 5 and avg_confidence > 0.7:
            approach = "Aggressive Management"
            intervention = "Consider interventional procedures (ESWL, URS, PCNL)"
            monitoring = "Close monitoring with frequent follow-up"
        elif num_stones >= 3:
            approach = "Active Management"
            intervention = "Consider medical expulsive therapy or intervention"
            monitoring = "Regular follow-up every 2-4 weeks"
        else:
            approach = "Conservative Management"
            intervention = "Lifestyle modifications, increased fluid intake, observation"
            monitoring = "Routine follow-up every 3-6 months"
        
        return {
            'management_approach': approach,
            'intervention_strategy': intervention,
            'monitoring_frequency': monitoring,
            'pain_management': "Consider NSAIDs or opiates for symptomatic stones",
            'prevention': "Advise on dietary modifications and hydration"
        }
    
    def _suggest_follow_up(self, detections: List[Dict]) -> Dict:
        """Suggest follow-up plan."""
        num_stones = len(detections)
        avg_confidence = np.mean([d['confidence'] for d in detections])
        
        if num_stones >= 5 or avg_confidence > 0.8:
            return {
                'frequency': '2 weeks',
                'studies': ['Repeat X-ray', 'Serum creatinine', 'Ultrasound'],
                'provider': 'Urologist or Nephrologist',
                'rationale': 'High stone burden or high confidence findings'
            }
        elif num_stones >= 3:
            return {
                'frequency': '1 month',
                'studies': ['Repeat X-ray', 'Basic metabolic panel'],
                'provider': 'Primary care with urology consult',
                'rationale': 'Moderate stone burden'
            }
        else:
            return {
                'frequency': '3-6 months',
                'studies': ['Repeat imaging if symptomatic'],
                'provider': 'Primary care',
                'rationale': 'Minimal stone burden'
            }
    
    def _identify_risk_factors(self, detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Identify clinical risk factors."""
        h, w = image_shape
        num_stones = len(detections)
        
        risk_factors = {
            'stone_count_risk': 'High' if num_stones >= 5 else 'Medium' if num_stones >= 3 else 'Low',
            'size_risk': self._assess_size_risk(detections, w, h),
            'location_risk': self._assess_location_risk(detections, w, h),
            'confidence_risk': self._assess_confidence_risk(detections),
            'overall_risk_level': 'Unknown'
        }
        
        # Overall risk calculation
        risk_score = 0
        if num_stones >= 5:
            risk_score += 30
        elif num_stones >= 3:
            risk_score += 20
        else:
            risk_score += 10
        
        avg_conf = np.mean([d['confidence'] for d in detections])
        if avg_conf > 0.8:
            risk_score += 30
        elif avg_conf > 0.6:
            risk_score += 20
        else:
            risk_score += 10
        
        if risk_score >= 50:
            overall_risk = 'High'
        elif risk_score >= 30:
            overall_risk = 'Moderate'
        else:
            overall_risk = 'Low'
        
        risk_factors['overall_risk_level'] = overall_risk
        risk_factors['risk_score'] = risk_score
        
        return risk_factors
    
    def _assess_size_risk(self, detections: List[Dict], w: int, h: int) -> str:
        """Assess risk based on stone size."""
        areas = [abs(b[2] - b[0]) * abs(b[3] - b[1]) for det in detections for b in [det['bbox']]]
        if not areas:
            return 'Unknown'
        
        max_area = max(areas)
        relative_size = max_area / (w * h)
        
        if relative_size > 0.02:  # More than 2% of image
            return 'High - Large stones may cause obstruction'
        elif relative_size > 0.01:
            return 'Medium - Moderate size stones'
        else:
            return 'Low - Small stones'
    
    def _assess_location_risk(self, detections: List[Dict], w: int, h: int) -> str:
        """Assess risk based on stone location."""
        kidney_count = 0
        bladder_count = 0
        
        for det in detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2 / w
            cy = (bbox[1] + bbox[3]) / 2 / h
            
            if cx < 0.35 or cx > 0.65:
                kidney_count += 1
            elif cy > 0.7:
                bladder_count += 1
        
        if kidney_count > 3:
            return 'High - Multiple kidney stones risk obstruction'
        elif kidney_count > 0:
            return 'Medium - Kidney stones require monitoring'
        elif bladder_count > 0:
            return 'Low-Medium - Bladder stones generally lower risk'
        else:
            return 'Low'
    
    def _assess_confidence_risk(self, detections: List[Dict]) -> str:
        """Assess risk based on confidence levels."""
        confidences = [d['confidence'] for d in detections]
        avg_conf = np.mean(confidences)
        
        if avg_conf > 0.8:
            return 'High - Very confident detections suggest definite pathology'
        elif avg_conf > 0.6:
            return 'Medium - Moderate confidence, likely true positive'
        else:
            return 'Low - Low confidence, verify with additional imaging'
    
    def _clinical_interpretation(self, detections: List[Dict]) -> str:
        """Provide clinical interpretation."""
        num_stones = len(detections)
        avg_conf = np.mean([d['confidence'] for d in detections])
        
        if num_stones >= 5 and avg_conf > 0.7:
            return """Multiple high-confidence kidney stone detections identified in this KUB X-ray. 
            The presence of numerous stones, particularly in the kidney regions, suggests significant 
            urolithiasis that may require intervention. Consider CT for detailed characterization and 
            urgent urologic consultation. Monitor for signs of obstruction or infection."""
        
        elif num_stones >= 3:
            return """Several kidney stones detected with moderate to high confidence. 
            These findings warrant clinical correlation with patient symptoms and laboratory studies. 
            Close follow-up recommended to assess stone progression and need for intervention."""
        
        elif num_stones >= 1:
            return """Kidney stone(s) detected in this study. Clinical correlation recommended 
            with patient symptoms, history, and laboratory evaluation. Routine follow-up imaging 
            may be appropriate depending on symptoms."""
        
        else:
            return "No definite kidney stones identified in this imaging study."
    
    def _no_detections_analysis(self) -> Dict:
        """Return analysis for case with no detections."""
        return {
            'severity_assessment': {
                'severity_level': 'None',
                'severity_score': 0,
                'interpretation': 'No kidney stones detected'
            },
            'treatment_urgency': {
                'urgency_level': TreatmentUrgency.ROUTINE.value,
                'recommended_time_frame': 'No immediate action needed'
            },
            'clinical_recommendations': {
                'primary_recommendations': [
                    {
                        'priority': 'Low',
                        'action': 'No intervention needed',
                        'rationale': 'No stones detected on current imaging'
                    }
                ]
            },
            'clinical_interpretation': 'No kidney stones identified in this X-ray. Findings are normal for kidney stone presence.'
        }
    
    def generate_clinical_report(self, detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Generate comprehensive clinical report."""
        relevance_analysis = self.analyze_clinical_relevance(detections, image_shape)
        
        report = {
            'executive_summary': self._generate_executive_summary(relevance_analysis),
            'findings_summary': {
                'total_stones': len(detections),
                'severity': relevance_analysis['severity_assessment']['severity_level'],
                'urgency': relevance_analysis['treatment_urgency']['urgency_level'],
                'primary_concern': relevance_analysis['anatomical_relevance']['primary_concern_location']
            },
            'detailed_analysis': relevance_analysis,
            'clinical_decision_support': self._clinical_decision_support(relevance_analysis)
        }
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict) -> str:
        """Generate executive summary."""
        severity = analysis['severity_assessment']['severity_level']
        urgency = analysis['treatment_urgency']['urgency_level']
        num_stones = analysis['severity_assessment']['num_stones']
        
        summary = f"""Detection Results:
- Stones Identified: {num_stones}
- Severity: {severity}
- Treatment Urgency: {urgency}
- Primary Recommendation: {analysis['clinical_recommendations']['primary_recommendations'][0]['action'] if analysis['clinical_recommendations']['primary_recommendations'] else 'Clinical correlation recommended'}
"""
        return summary
    
    def _clinical_decision_support(self, analysis: Dict) -> Dict:
        """Provide clinical decision support."""
        severity = analysis['severity_assessment']['severity_level']
        urgency = analysis['treatment_urgency']['urgency_level']
        
        decision_tree = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'referral_needed': False,
            'imaging_additional': False
        }
        
        if severity in ['Severe', 'Critical'] or urgency == TreatmentUrgency.EMERGENCY.value:
            decision_tree['immediate_actions'] = [
                'Urgent urologic consultation',
                'Pain management assessment',
                'Renal function evaluation',
                'Consider additional imaging (CT)'
            ]
            decision_tree['referral_needed'] = True
            decision_tree['imaging_additional'] = True
        
        elif severity == 'Moderate' or urgency == TreatmentUrgency.URGENT.value:
            decision_tree['short_term_actions'] = [
                'Urology referral within 1 week',
                'Baseline laboratory studies',
                'Patient education on stones',
                'Consider expulsive therapy'
            ]
            decision_tree['referral_needed'] = True
        
        else:
            decision_tree['long_term_actions'] = [
                'Primary care follow-up',
                'Lifestyle counseling',
                'Increased hydration',
                'Repeat imaging in 3-6 months'
            ]
        
        return decision_tree

