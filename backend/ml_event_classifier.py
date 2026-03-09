# New file: ml_event_classifier.py

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np
import os
from modules.event_detector import GameEvent

class EventClassifier(nn.Module):
    """
    LSTM-based event classifier for football actions.
    
    Input: Temporal sequence of player/ball features
    Output: Event probabilities
    """
    
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=18):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=0.2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep
        final = attended[:, -1, :]
        
        # Classify
        logits = self.classifier(final)
        return logits

class MLEventDetector:
    """ML-based event detection using trained neural network"""
    
    def __init__(self, model_path: str = None, config=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = EventClassifier().to(self.device)
        self.window_size = 16  # frames
        self.feature_buffer = []
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        
        # Event labels
        self.event_classes = [
            "pass", "shot", "dribble", "tackle", "clearance",
            "block", "interception", "pressure", "carry",
            "ball_recovery", "duel", "miscontrol", "dispossessed",
            "goalkeeper_save", "foul", "set_piece", "sprint", "other"
        ]
    
    def extract_features(self, player_tracks, ball_track, possessor_id) -> np.ndarray:
        """
        Extract features for current frame.
        
        Features include:
        - Ball position and velocity
        - Possessor position and velocity
        - Nearby players' positions and teams
        - Spatial relationships
        """
        features = np.zeros(64)
        
        if ball_track:
            features[0:2] = ball_track.center
            features[2] = ball_track.velocity
        
        if possessor_id and possessor_id in player_tracks:
            poss = player_tracks[possessor_id]
            features[3:5] = poss.center
            features[5] = poss.velocity
            features[6] = poss.team_id
            
            # Find nearby players
            nearby = []
            for tid, track in player_tracks.items():
                if tid != possessor_id:
                    dist = np.linalg.norm(
                        np.array(track.center) - np.array(poss.center)
                    )
                    if dist < 150:
                        nearby.append((dist, track))
            
            # Sort by distance
            nearby.sort(key=lambda x: x[0])
            
            # Encode up to 5 nearest players
            for i, (dist, track) in enumerate(nearby[:5]):
                idx = 7 + i * 5
                features[idx] = track.center[0]
                features[idx+1] = track.center[1]
                features[idx+2] = track.team_id
                features[idx+3] = track.velocity
                features[idx+4] = dist
        
        return features
    
    def detect_events(self, player_tracks, ball_track, possessor_id, 
                      frame_idx, fps) -> List[GameEvent]:
        """Detect events using ML model"""
        
        # Extract features
        features = self.extract_features(player_tracks, ball_track, possessor_id)
        self.feature_buffer.append(features)
        
        # Keep sliding window
        if len(self.feature_buffer) > self.window_size:
            self.feature_buffer.pop(0)
        
        # Need full window for prediction
        if len(self.feature_buffer) < self.window_size:
            return []
        
        # Prepare input
        x = np.array(self.feature_buffer).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
        
        # Threshold
        conf_threshold = 0.7
        if confidences[0] > conf_threshold:
            event_type = self.event_classes[predictions[0]]
            
            # Skip "other" class
            if event_type != "other":
                return [GameEvent(
                    event_type=event_type,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    confidence=float(confidences[0]),
                    position=(features[0], features[1]),
                    player_id=possessor_id,
                    source="ml"
                )]
        
        return []