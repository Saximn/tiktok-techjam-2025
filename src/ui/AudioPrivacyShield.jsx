/**
 * VoiceShield Privacy Dashboard - Lynx UI Component
 * Real-time voice privacy visualization for TikTok Live streaming
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Animated,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { LynxCore, LynxAnimations, LynxAudio } from '@lynx-ui/core';
import { VoiceSpectrogram } from './VoiceSpectrogram';
import { PrivacySlider } from './PrivacySlider';
import { SpeakerMap } from './SpeakerMap';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

export const AudioPrivacyShield = ({ 
  isStreaming = false,
  viewerCount = 0,
  privacyLevel = 0.6,
  onPrivacyLevelChange,
  onEmergencyStop,
  realTimeData = null 
}) => {
  const [audioMetrics, setAudioMetrics] = useState({
    amplitude: 0,
    frequency: 440,
    piiDetected: false,
    backgroundVoices: 0,
    protectionActive: true
  });

  const [privacyAlerts, setPrivacyAlerts] = useState([]);
  const privacyHaloAnimation = useRef(new Animated.Value(0)).current;
  const pulseAnimation = useRef(new Animated.Value(1)).current;
  const emergencyAnimation = useRef(new Animated.Value(0)).current;

  // Real-time audio data processing
  useEffect(() => {
    if (isStreaming && realTimeData) {
      setAudioMetrics(prev => ({
        ...prev,
        amplitude: realTimeData.amplitude || 0,
        frequency: realTimeData.frequency || 440,
        piiDetected: realTimeData.piiDetected || false,
        backgroundVoices: realTimeData.backgroundVoices || 0,
        protectionActive: realTimeData.protectionActive !== false
      }));

      // Trigger privacy alerts
      if (realTimeData.piiDetected) {
        addPrivacyAlert('PII detected and masked', 'warning');
      }
      if (realTimeData.backgroundVoices > 0) {
        addPrivacyAlert(`${realTimeData.backgroundVoices} background voice(s) filtered`, 'info');
      }
    }
  }, [realTimeData, isStreaming]);

  // Voice Privacy Halo Animation
  useEffect(() => {
    const animateHalo = () => {
      Animated.loop(
        Animated.sequence([
          Animated.timing(privacyHaloAnimation, {
            toValue: 1,
            duration: 2000,
            useNativeDriver: true,
          }),
          Animated.timing(privacyHaloAnimation, {
            toValue: 0,
            duration: 2000,
            useNativeDriver: true,
          }),
        ])
      ).start();
    };

    if (audioMetrics.protectionActive) {
      animateHalo();
    }
  }, [audioMetrics.protectionActive, privacyHaloAnimation]);

  // Protection level pulse
  useEffect(() => {
    const pulse = () => {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnimation, {
            toValue: 1.1,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnimation, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      ).start();
    };

    pulse();
  }, [pulseAnimation]);

  const addPrivacyAlert = (message, type) => {
    const alert = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setPrivacyAlerts(prev => [...prev.slice(-4), alert]); // Keep last 5 alerts
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      setPrivacyAlerts(prev => prev.filter(a => a.id !== alert.id));
    }, 5000);
  };

  const handleEmergencyStop = () => {
    // Emergency animation
    Animated.sequence([
      Animated.timing(emergencyAnimation, {
        toValue: 1,
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(emergencyAnimation, {
        toValue: 0,
        duration: 200,
        useNativeDriver: true,
      }),
    ]).start();

    onEmergencyStop && onEmergencyStop();
    addPrivacyAlert('Emergency privacy activated', 'emergency');
  };

  const getPrivacyColor = () => {
    if (privacyLevel >= 0.8) return '#00ff88'; // High protection - green
    if (privacyLevel >= 0.5) return '#ffaa00'; // Medium protection - orange  
    return '#ff4444'; // Low protection - red
  };

  return (
    <LynxCore.Container style={styles.container}>
      {/* Voice Privacy Halo */}
      <Animated.View 
        style={[
          styles.privacyHalo,
          {
            opacity: privacyHaloAnimation,
            transform: [{
              scale: privacyHaloAnimation.interpolate({
                inputRange: [0, 1],
                outputRange: [0.8, 1.2]
              })
            }],
            borderColor: getPrivacyColor(),
          }
        ]}
      />

      {/* Real-Time Audio Visualization */}
      <View style={styles.audioVisualizationContainer}>
        <VoiceSpectrogram
          realtime={true}
          privacyMask={audioMetrics.protectionActive}
          amplitude={audioMetrics.amplitude}
          frequency={audioMetrics.frequency}
          style={styles.spectrogram}
        />
        
        {/* Privacy Protection Overlay */}
        {audioMetrics.protectionActive && (
          <Animated.View 
            style={[
              styles.protectionOverlay,
              { 
                opacity: privacyLevel * 0.3,
                backgroundColor: getPrivacyColor(),
                transform: [{ scale: pulseAnimation }]
              }
            ]}
          />
        )}
      </View>

      {/* Privacy Control Slider */}
      <View style={styles.controlsContainer}>
        <Text style={styles.controlLabel}>Privacy Protection Level</Text>
        <PrivacySlider
          value={privacyLevel}
          onChange={onPrivacyLevelChange}
          visualFeedback="waveform-morphing"
          style={styles.privacySlider}
        />
        <Text style={styles.privacyPercentage}>
          {Math.round(privacyLevel * 100)}%
        </Text>
      </View>

      {/* Speaker Detection Map */}
      <SpeakerMap
        speakers={[
          { id: 'main', name: 'You', privacyLevel: privacyLevel, active: true },
          ...(audioMetrics.backgroundVoices > 0 ? 
            Array.from({ length: audioMetrics.backgroundVoices }, (_, i) => ({
              id: `bg_${i}`,
              name: `Background ${i + 1}`,
              privacyLevel: 1.0,
              active: true,
              filtered: true
            })) : [])
        ]}
        privacyStatus="individual"
        style={styles.speakerMap}
      />

      {/* Live Stream Info */}
      <View style={styles.streamInfo}>
        <View style={styles.streamIndicator}>
          <View style={[styles.liveIndicator, { backgroundColor: isStreaming ? '#ff0000' : '#666' }]} />
          <Text style={styles.liveText}>
            {isStreaming ? 'LIVE' : 'OFFLINE'}
          </Text>
        </View>
        
        <Text style={styles.viewerCount}>
          👥 {viewerCount.toLocaleString()} viewers
        </Text>
      </View>

      {/* Emergency Privacy Button */}
      <TouchableOpacity 
        style={[styles.emergencyButton, { backgroundColor: getPrivacyColor() }]}
        onPress={handleEmergencyStop}
        activeOpacity={0.8}
      >
        <Animated.View 
          style={[
            styles.emergencyButtonInner,
            {
              transform: [{
                scale: emergencyAnimation.interpolate({
                  inputRange: [0, 1],
                  outputRange: [1, 1.2]
                })
              }]
            }
          ]}
        >
          <Text style={styles.emergencyButtonText}>🛡️</Text>
          <Text style={styles.emergencyButtonLabel}>Emergency Privacy</Text>
        </Animated.View>
      </TouchableOpacity>

      {/* Privacy Alerts */}
      <View style={styles.alertsContainer}>
        {privacyAlerts.map(alert => (
          <LynxAnimations.FadeInUp
            key={alert.id}
            duration={300}
            style={[
              styles.alert,
              { backgroundColor: getAlertColor(alert.type) }
            ]}
          >
            <Text style={styles.alertText}>{alert.message}</Text>
            <Text style={styles.alertTime}>{alert.timestamp}</Text>
          </LynxAnimations.FadeInUp>
        ))}
      </View>
    </LynxCore.Container>
  );
};

const getAlertColor = (type) => {
  switch (type) {
    case 'emergency': return 'rgba(255, 68, 68, 0.9)';
    case 'warning': return 'rgba(255, 170, 0, 0.9)';
    case 'info': return 'rgba(0, 255, 136, 0.9)';
    default: return 'rgba(255, 255, 255, 0.9)';
  }
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
    padding: 20,
    alignItems: 'center',
  },
  privacyHalo: {
    position: 'absolute',
    width: screenWidth * 0.8,
    height: screenWidth * 0.8,
    borderRadius: screenWidth * 0.4,
    borderWidth: 3,
    top: '10%',
  },
  audioVisualizationContainer: {
    width: '90%',
    height: 200,
    marginTop: 60,
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: '#1a1a1a',
  },
  spectrogram: {
    width: '100%',
    height: '100%',
  },
  protectionOverlay: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 20,
  },
  controlsContainer: {
    width: '90%',
    marginTop: 30,
    alignItems: 'center',
  },
  controlLabel: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 10,
  },
  privacySlider: {
    width: '100%',
    height: 40,
  },
  privacyPercentage: {
    color: '#ffffff',
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 10,
  },
  speakerMap: {
    width: '90%',
    marginTop: 20,
  },
  streamInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '90%',
    marginTop: 20,
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: '#1a1a1a',
    borderRadius: 15,
  },
  streamIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  liveIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  liveText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  viewerCount: {
    color: '#ffffff',
    fontSize: 16,
  },
  emergencyButton: {
    position: 'absolute',
    bottom: 30,
    right: 30,
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  emergencyButtonInner: {
    alignItems: 'center',
  },
  emergencyButtonText: {
    fontSize: 28,
  },
  emergencyButtonLabel: {
    color: '#ffffff',
    fontSize: 10,
    fontWeight: 'bold',
    textAlign: 'center',
    marginTop: 2,
  },
  alertsContainer: {
    position: 'absolute',
    top: 20,
    right: 20,
    width: '40%',
  },
  alert: {
    padding: 12,
    borderRadius: 10,
    marginBottom: 8,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  alertText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
  },
  alertTime: {
    color: '#ffffff',
    fontSize: 10,
    opacity: 0.8,
    marginTop: 4,
  },
});

export default AudioPrivacyShield;
