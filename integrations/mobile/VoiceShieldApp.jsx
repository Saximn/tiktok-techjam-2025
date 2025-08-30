/**
 * VoiceShield Mobile App - TikTok Live Privacy Protection
 * Main app component integrating all privacy features
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StatusBar,
  SafeAreaView,
  StyleSheet,
  Alert,
  Dimensions,
} from 'react-native';
import { LynxCore, LynxNavigation } from '@lynx-ui/mobile';
import AudioPrivacyShield from '../ui/AudioPrivacyShield';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const VoiceShieldApp = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [privacyLevel, setPrivacyLevel] = useState(0.6);
  const [viewerCount, setViewerCount] = useState(0);
  const [currentMode, setCurrentMode] = useState('personal'); // personal, meeting, public, emergency
  const [realTimeAudioData, setRealTimeAudioData] = useState(null);
  const [streamMetrics, setStreamMetrics] = useState({
    duration: 0,
    piiBlocked: 0,
    backgroundFiltered: 0,
    privacyAlerts: 0
  });

  // Simulate real-time audio data
  const audioSimulationRef = useRef(null);

  useEffect(() => {
    if (isStreaming) {
      startAudioSimulation();
    } else {
      stopAudioSimulation();
    }
    
    return () => stopAudioSimulation();
  }, [isStreaming]);

  const startAudioSimulation = () => {
    audioSimulationRef.current = setInterval(() => {
      // Generate realistic audio metrics
      const amplitude = Math.random() * 0.8 + 0.1; // 0.1 to 0.9
      const frequency = 200 + Math.random() * 300; // 200-500 Hz
      const piiDetected = Math.random() < 0.02; // 2% chance of PII
      const backgroundVoices = Math.random() < 0.1 ? Math.floor(Math.random() * 3) + 1 : 0;
      
      setRealTimeAudioData({
        amplitude,
        frequency,
        piiDetected,
        backgroundVoices,
        protectionActive: true
      });

      // Update viewer count realistically
      if (isStreaming) {
        setViewerCount(prev => {
          const change = Math.floor((Math.random() - 0.5) * 20);
          return Math.max(0, prev + change);
        });
      }
    }, 100); // 10fps updates
  };

  const stopAudioSimulation = () => {
    if (audioSimulationRef.current) {
      clearInterval(audioSimulationRef.current);
      audioSimulationRef.current = null;
    }
  };

  const handleStartStream = () => {
    Alert.alert(
      "Start TikTok Live Stream?",
      "VoiceShield will protect your privacy during streaming",
      [
        { text: "Cancel", style: "cancel" },
        { 
          text: "Go Live", 
          onPress: () => {
            setIsStreaming(true);
            setViewerCount(Math.floor(Math.random() * 50) + 10); // Start with 10-60 viewers
            
            // Show success message
            setTimeout(() => {
              Alert.alert(
                "🔴 Live Stream Started!",
                "VoiceShield is actively protecting your privacy",
                [{ text: "OK" }]
              );
            }, 1000);
          }
        }
      ]
    );
  };

  const handleStopStream = () => {
    Alert.alert(
      "End Live Stream?",
      `Stream duration: ${Math.floor(streamMetrics.duration / 60)}m ${streamMetrics.duration % 60}s\nPrivacy events handled: ${streamMetrics.privacyAlerts}`,
      [
        { text: "Keep Streaming", style: "cancel" },
        { 
          text: "End Stream", 
          style: "destructive",
          onPress: () => {
            setIsStreaming(false);
            setViewerCount(0);
            setRealTimeAudioData(null);
          }
        }
      ]
    );
  };

  const handlePrivacyLevelChange = (newLevel) => {
    setPrivacyLevel(newLevel);
    
    // Update privacy mode based on level
    if (newLevel >= 0.8) {
      setCurrentMode('public');
    } else if (newLevel >= 0.5) {
      setCurrentMode('meeting');
    } else {
      setCurrentMode('personal');
    }
  };

  const handleEmergencyStop = () => {
    Alert.alert(
      "🚨 Emergency Privacy Activated",
      "Audio stream has been immediately muted. Your privacy is protected.",
      [
        { text: "Keep Muted", style: "default" },
        { text: "Resume Stream", style: "cancel", onPress: () => {
          // Resume normal operation
        }}
      ]
    );
  };

  const getPrivacyModeColor = () => {
    switch (currentMode) {
      case 'emergency': return '#ff0000';
      case 'public': return '#00ff88';
      case 'meeting': return '#ffaa00';
      case 'personal': return '#4488ff';
      default: return '#666666';
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.titleContainer}>
          <Text style={styles.appTitle}>VoiceShield</Text>
          <Text style={styles.appSubtitle}>TikTok Live Privacy Protection</Text>
        </View>
        
        <View style={[styles.modeIndicator, { backgroundColor: getPrivacyModeColor() }]}>
          <Text style={styles.modeText}>{currentMode.toUpperCase()}</Text>
        </View>
      </View>

      {/* Main Privacy Dashboard */}
      <View style={styles.dashboardContainer}>
        <AudioPrivacyShield
          isStreaming={isStreaming}
          viewerCount={viewerCount}
          privacyLevel={privacyLevel}
          onPrivacyLevelChange={handlePrivacyLevelChange}
          onEmergencyStop={handleEmergencyStop}
          realTimeData={realTimeAudioData}
        />
      </View>

      {/* Stream Controls */}
      <View style={styles.controlsContainer}>
        {!isStreaming ? (
          <TouchableOpacity 
            style={styles.streamButton} 
            onPress={handleStartStream}
          >
            <Text style={styles.streamButtonText}>🔴 Go Live on TikTok</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity 
            style={[styles.streamButton, styles.stopStreamButton]} 
            onPress={handleStopStream}
          >
            <Text style={styles.streamButtonText}>⏹️ End Stream</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Quick Settings */}
      <View style={styles.quickSettings}>
        <TouchableOpacity 
          style={styles.quickSetting}
          onPress={() => handlePrivacyLevelChange(0.3)}
        >
          <Text style={styles.quickSettingIcon}>👨‍👩‍👧‍👦</Text>
          <Text style={styles.quickSettingText}>Personal</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.quickSetting}
          onPress={() => handlePrivacyLevelChange(0.7)}
        >
          <Text style={styles.quickSettingIcon}>💼</Text>
          <Text style={styles.quickSettingText}>Meeting</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.quickSetting}
          onPress={() => handlePrivacyLevelChange(1.0)}
        >
          <Text style={styles.quickSettingIcon}>🌍</Text>
          <Text style={styles.quickSettingText}>Public</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.quickSetting}
          onPress={handleEmergencyStop}
        >
          <Text style={styles.quickSettingIcon}>🚨</Text>
          <Text style={styles.quickSettingText}>Emergency</Text>
        </TouchableOpacity>
      </View>

      {/* Stream Stats */}
      {isStreaming && (
        <View style={styles.statsContainer}>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{viewerCount}</Text>
            <Text style={styles.statLabel}>Viewers</Text>
          </View>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{Math.round(privacyLevel * 100)}%</Text>
            <Text style={styles.statLabel}>Protection</Text>
          </View>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{streamMetrics.piiBlocked}</Text>
            <Text style={styles.statLabel}>PII Blocked</Text>
          </View>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{streamMetrics.backgroundFiltered}</Text>
            <Text style={styles.statLabel}>BG Filtered</Text>
          </View>
        </View>
      )}

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          {isStreaming ? 
            "🛡️ Privacy protection active • Real-time voice anonymization" :
            "Ready to protect your privacy on TikTok Live"
          }
        </Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#333333',
  },
  titleContainer: {
    flex: 1,
  },
  appTitle: {
    color: '#ffffff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  appSubtitle: {
    color: '#aaaaaa',
    fontSize: 12,
    marginTop: 2,
  },
  modeIndicator: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  modeText: {
    color: '#ffffff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  dashboardContainer: {
    flex: 1,
    margin: 10,
  },
  controlsContainer: {
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  streamButton: {
    backgroundColor: '#ff0050',
    paddingVertical: 15,
    borderRadius: 25,
    alignItems: 'center',
    elevation: 3,
    shadowColor: '#ff0050',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  stopStreamButton: {
    backgroundColor: '#666666',
  },
  streamButtonText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  quickSettings: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
    paddingBottom: 15,
  },
  quickSetting: {
    alignItems: 'center',
    padding: 10,
  },
  quickSettingIcon: {
    fontSize: 24,
    marginBottom: 5,
  },
  quickSettingText: {
    color: '#aaaaaa',
    fontSize: 10,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#1a1a1a',
    marginHorizontal: 20,
    borderRadius: 15,
    marginBottom: 10,
  },
  stat: {
    alignItems: 'center',
  },
  statValue: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  statLabel: {
    color: '#aaaaaa',
    fontSize: 10,
    marginTop: 2,
  },
  footer: {
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderTopWidth: 1,
    borderTopColor: '#333333',
    alignItems: 'center',
  },
  footerText: {
    color: '#aaaaaa',
    fontSize: 12,
    textAlign: 'center',
  },
});

export default VoiceShieldApp;
