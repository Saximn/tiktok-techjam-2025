/**
 * SpeakerMap - Multi-speaker privacy visualization and control
 * Shows individual privacy protection status for each detected speaker
 */

import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import Svg, { Circle, Path, Defs, RadialGradient, Stop } from 'react-native-svg';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withTiming,
  interpolate,
  interpolateColor
} from 'react-native-reanimated';
import { LynxAnimations } from '@lynx-ui/core';

export const SpeakerMap = ({
  speakers = [],
  privacyStatus = 'individual',
  onSpeakerToggle,
  style,
  width = 300,
  height = 150
}) => {
  return (
    <View style={[styles.container, style, { width, height }]}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Speaker Privacy Map</Text>
        <Text style={styles.subtitle}>
          {speakers.filter(s => s.active).length} active • {privacyStatus} protection
        </Text>
      </View>

      {/* Speaker Grid */}
      <View style={styles.speakerGrid}>
        {speakers.map((speaker, index) => (
          <SpeakerNode
            key={speaker.id}
            speaker={speaker}
            index={index}
            totalSpeakers={speakers.length}
            onToggle={() => onSpeakerToggle && onSpeakerToggle(speaker.id)}
          />
        ))}
        
        {speakers.length === 0 && (
          <View style={styles.emptyState}>
            <Text style={styles.emptyText}>🎤</Text>
            <Text style={styles.emptyLabel}>No speakers detected</Text>
          </View>
        )}
      </View>

      {/* Privacy Legend */}
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: '#00ff88' }]} />
          <Text style={styles.legendText}>Protected</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: '#ffaa00' }]} />
          <Text style={styles.legendText}>Filtering</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: '#ff4444' }]} />
          <Text style={styles.legendText}>Exposed</Text>
        </View>
      </View>
    </View>
  );
};

const SpeakerNode = ({ speaker, index, totalSpeakers, onToggle }) => {
  const scale = useSharedValue(1);
  const pulseOpacity = useSharedValue(0.5);
  const protectionRadius = useSharedValue(speaker.privacyLevel * 40);

  // Initialize animations
  useEffect(() => {
    scale.value = withSpring(1, { delay: index * 100 });
    
    if (speaker.active) {
      pulseOpacity.value = withRepeat(
        withTiming(1, { duration: 1000 }),
        -1,
        true
      );
    }
    
    protectionRadius.value = withSpring(speaker.privacyLevel * 40);
  }, [speaker.active, speaker.privacyLevel, index]);

  // Speaker position in circular layout
  const getPosition = () => {
    if (totalSpeakers === 1) {
      return { x: 150, y: 75 }; // Center
    }
    
    const angle = (index / totalSpeakers) * 2 * Math.PI - (Math.PI / 2);
    const radius = 60;
    const centerX = 150;
    const centerY = 75;
    
    return {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius
    };
  };

  const position = getPosition();

  const getPrivacyColor = () => {
    if (speaker.privacyLevel >= 0.8) return '#00ff88'; // High protection
    if (speaker.privacyLevel >= 0.5) return '#ffaa00'; // Medium protection
    return '#ff4444'; // Low/no protection
  };

  const nodeStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: position.x - 25 },
      { translateY: position.y - 25 },
      { scale: scale.value }
    ],
  }));

  const pulseStyle = useAnimatedStyle(() => ({
    opacity: speaker.active ? pulseOpacity.value * 0.6 : 0,
    transform: [
      { scale: interpolate(pulseOpacity.value, [0, 1], [1, 1.3]) }
    ],
  }));

  return (
    <TouchableOpacity onPress={onToggle} activeOpacity={0.8}>
      <Animated.View style={[styles.speakerNode, nodeStyle]}>
        {/* Privacy Protection Ring */}
        <Animated.View style={[styles.protectionRing, pulseStyle]}>
          <Svg width="60" height="60">
            <Defs>
              <RadialGradient id={`gradient_${speaker.id}`} cx="50%" cy="50%">
                <Stop offset="0%" stopColor={getPrivacyColor()} stopOpacity="0.8" />
                <Stop offset="100%" stopColor={getPrivacyColor()} stopOpacity="0.2" />
              </RadialGradient>
            </Defs>
            <Circle
              cx="30"
              cy="30"
              r="28"
              fill={`url(#gradient_${speaker.id})`}
            />
          </Svg>
        </Animated.View>

        {/* Speaker Avatar */}
        <View style={[styles.speakerAvatar, { borderColor: getPrivacyColor() }]}>
          <Text style={styles.speakerIcon}>
            {speaker.name === 'You' ? '👤' : 
             speaker.filtered ? '🔇' : 
             speaker.id.includes('bg_') ? '👥' : '🎤'}
          </Text>
          
          {/* Privacy Level Indicator */}
          <View style={[styles.privacyIndicator, { backgroundColor: getPrivacyColor() }]}>
            <Text style={styles.privacyLevel}>
              {Math.round(speaker.privacyLevel * 100)}%
            </Text>
          </View>
        </View>

        {/* Speaker Label */}
        <View style={styles.speakerLabel}>
          <Text style={styles.speakerName}>{speaker.name}</Text>
          <Text style={styles.speakerStatus}>
            {speaker.filtered ? 'Filtered' : 
             speaker.active ? 'Active' : 'Inactive'}
          </Text>
        </View>

        {/* Connection Lines to Main Speaker */}
        {speaker.name !== 'You' && (
          <Svg 
            style={styles.connectionLine} 
            width="300" 
            height="150"
          >
            <Path
              d={`M ${position.x} ${position.y} Q 150 75 150 75`}
              stroke={getPrivacyColor()}
              strokeWidth="1"
              fill="none"
              strokeDasharray="5,5"
              opacity="0.4"
            />
          </Svg>
        )}

        {/* Privacy Features Icons */}
        <View style={styles.featuresContainer}>
          {speaker.privacyLevel > 0.7 && (
            <LynxAnimations.FadeIn delay={200}>
              <Text style={styles.featureIcon}>🔒</Text>
            </LynxAnimations.FadeIn>
          )}
          {speaker.filtered && (
            <LynxAnimations.FadeIn delay={300}>
              <Text style={styles.featureIcon}>🎚️</Text>
            </LynxAnimations.FadeIn>
          )}
          {speaker.privacyLevel > 0.5 && (
            <LynxAnimations.FadeIn delay={400}>
              <Text style={styles.featureIcon}>🛡️</Text>
            </LynxAnimations.FadeIn>
          )}
        </View>
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: 15,
    padding: 15,
    position: 'relative',
  },
  header: {
    alignItems: 'center',
    marginBottom: 15,
  },
  title: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  subtitle: {
    color: '#aaaaaa',
    fontSize: 12,
    marginTop: 2,
  },
  speakerGrid: {
    flex: 1,
    position: 'relative',
  },
  speakerNode: {
    position: 'absolute',
    width: 50,
    height: 50,
    alignItems: 'center',
  },
  protectionRing: {
    position: 'absolute',
    width: 60,
    height: 60,
    borderRadius: 30,
    top: -5,
    left: -5,
  },
  speakerAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#2a2a2a',
    borderWidth: 2,
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  speakerIcon: {
    fontSize: 18,
  },
  privacyIndicator: {
    position: 'absolute',
    bottom: -5,
    right: -5,
    width: 16,
    height: 16,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  privacyLevel: {
    color: '#ffffff',
    fontSize: 8,
    fontWeight: 'bold',
  },
  speakerLabel: {
    alignItems: 'center',
    marginTop: 5,
  },
  speakerName: {
    color: '#ffffff',
    fontSize: 10,
    fontWeight: '600',
  },
  speakerStatus: {
    color: '#aaaaaa',
    fontSize: 8,
  },
  connectionLine: {
    position: 'absolute',
    top: 0,
    left: 0,
    zIndex: -1,
  },
  featuresContainer: {
    position: 'absolute',
    top: -10,
    right: -10,
    flexDirection: 'row',
  },
  featureIcon: {
    fontSize: 10,
    marginLeft: 2,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 30,
    opacity: 0.5,
  },
  emptyLabel: {
    color: '#666666',
    fontSize: 12,
    marginTop: 5,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#333333',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 5,
  },
  legendText: {
    color: '#aaaaaa',
    fontSize: 10,
  },
});

export default SpeakerMap;
