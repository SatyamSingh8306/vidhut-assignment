#!/usr/bin/env python
"""
Test script for the full multimodal monitoring pipeline.
Run with: python test_pipeline.py
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.detectors.facial_expression import FacialExpressionClassifier
from app.detectors.hand_gesture import HandGestureClassifier
from app.detectors.visual_speech import VisualSpeechClassifier
from app.detectors.isl_classifier import ISLSequenceRecognizer
from app.detectors.help_symbol import HelpSymbolDetector
from app.fusion.fusion_engine import AdaptiveFusionEngine, TemporalContextModel, UncertaintyEstimator, MultimodalSignal
from app.personalization.patient_personalization import PatientPersonalization
# VLMReasoner is optional - requires GPU and large model download
# from app.vlm.vlm_reasoner import VLMReasoner

def test_detectors():
    """Test all detectors with webcam."""
    print("=" * 60)
    print("Testing Multimodal Pipeline with Webcam")
    print("=" * 60)
    
    # Initialize detectors
    print("\nInitializing detectors...")
    face = FacialExpressionClassifier()
    hand = HandGestureClassifier(use_isl=True)
    speech = VisualSpeechClassifier()
    help_det = HelpSymbolDetector(confirm_frames=3)
    fusion = AdaptiveFusionEngine()
    temporal = TemporalContextModel()
    uncertainty = UncertaintyEstimator()
    personalization = PatientPersonalization()
    # vlm = VLMReasoner()  # Optional - requires GPU and large model download
    
    print("[OK] All detectors initialized")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return
    
    print("\nStarting webcam test. Press 'q' to quit.")
    print("Try making gestures: open palm, thumbs up, pointing, etc.")
    print("Try facial expressions: smile, frown, surprise, etc.")
    
    frame_count = 0
    patient_id = "test_patient"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = time.time()
        timestamp_ms = int(timestamp * 1000)
        
        # Run detectors
        expr, expr_conf, expr_confirmed = face.update(frame, timestamp_ms)
        gest, gest_conf, gest_confirmed = hand.update(frame, timestamp_ms)
        isl_intent, isl_conf = hand.get_isl_intent()
        sp, sp_conf, sp_confirmed = speech.update(frame, timestamp_ms)
        
        # Help detection (simplified - no pose)
        help_symbol, help_conf, help_req = help_det.update(None, frame=frame)
        
        # Prepare signals for fusion
        signals = []
        if expr != "unknown":
            signals.append(MultimodalSignal("facial_expression", expr_conf, timestamp))
        if gest != "unknown":
            signals.append(MultimodalSignal("hand_gesture", gest_conf, timestamp))
        if sp != "unknown":
            signals.append(MultimodalSignal("visual_speech", sp_conf, timestamp))
        if help_req:
            signals.append(MultimodalSignal("pose_body", help_conf, timestamp))
        
        # Run fusion
        fusion.update(*signals)
        
        intent_categories = {}
        if expr != "unknown":
            intent_categories[f"expr_{expr}"] = expr_conf
        if isl_intent != "unknown":
            intent_categories[f"isl_{isl_intent}"] = isl_conf
        elif gest != "unknown":
            intent_categories[f"gest_{gest}"] = gest_conf
        if sp != "unknown":
            intent_categories[f"speech_{sp}"] = sp_conf
        if help_req:
            intent_categories["help"] = help_conf
        
        fused_intent, fused_conf, breakdown = fusion.fuse(intent_categories)
        contextual_intent, contextual_conf = temporal.get_contextual_intent(fused_intent, fused_conf, patient_id)
        uncertainty_result = uncertainty.evaluate(signals)
        
        # Update temporal context
        temporal.update(contextual_intent, contextual_conf, timestamp)
        
        # Update personalization
        if contextual_intent != "unknown":
            personalization.learn_from_detection(patient_id, {contextual_intent: contextual_conf})
        
        # Display results on frame
        y_offset = 30
        line_height = 25
        
        cv2.putText(frame, f"Facial: {expr} ({expr_conf:.2f})", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"Hand: {gest} ({gest_conf:.2f})", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"ISL Intent: {isl_intent} ({isl_conf:.2f})", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"Speech: {sp} ({sp_conf:.2f})", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"Fused: {contextual_intent} ({contextual_conf:.2f})", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"Uncertainty: {uncertainty_result['overall_uncertainty']:.2f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y_offset += line_height
        
        if help_req:
            cv2.putText(frame, "*** HELP DETECTED ***", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        # Show FPS
        cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Multimodal Pipeline Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face.close()
    hand.close()
    speech.close()
    # vlm.close()
    
    print("\n[OK] Test completed")

if __name__ == "__main__":
    test_detectors()