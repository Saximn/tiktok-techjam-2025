/**
 * PrivacySlider - Interactive privacy level control with waveform morphing
 * Visual feedback shows how privacy protection affects audio processing
 */

import React, { useEffect, useRef } from 'react';
import { View, PanGestureHandler, StyleSheet } from 'react-native';
import Svg, { Path, Defs, LinearGradient, Stop, Circle } from 'react-native-svg';
import Animated, {
  useAnimatedGestureHandler,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  interpolate,
  runOnJS,
  interpolateColor
} from 'react-native-reanimated';
import { LynxCore, LynxAnimations } from '@lynx-ui/core';

export const PrivacySlider = ({
  value = 0.6,
  onChange,
  visualFeedback = 'waveform-morphing',
  style,
  width = 300,
  height = 60
}) => {
  const sliderWidth = width - 40; // Account for thumb size
  const thumbPosition = useSharedValue(value * sliderWidth);
  const waveformData = useRef(generateWaveformData()).current;
  const isDragging = useSharedValue(false);
  
  // Update thumb position when value prop changes
  useEffect(() => {
    thumbPosition.value = withSpring(value * sliderWidth);
  }, [value, sliderWidth]);

  // Generate sample waveform data
  function generateWaveformData() {
    return Array.from({ length: 100 }, (_, i) => {
      const x = (i / 99) * 2 * Math.PI;
      return Math.sin(x) + Math.sin(x * 3) * 0.3 + Math.sin(x * 5) * 0.1;
    });
  }

  // Pan gesture handler for slider interaction
  const panGestureHandler = useAnimatedGestureHandler({
    onStart: () => {
      isDragging.value = true;
    },
    onActive: (event) => {
      const newPosition = Math.max(0, Math.min(sliderWidth, event.x - 20));
      thumbPosition.value = newPosition;
      
      const newValue = newPosition / sliderWidth;
      runOnJS(onChange)(newValue);
    },
    onEnd: () => {
      isDragging.value = false;
    },
  });

  // Generate waveform path with privacy morphing effect
  const generateWaveformPath = (privacyLevel) => {
    const points = [];
    const baselineY = height * 0.5;
    
    waveformData.forEach((amplitude, index) => {
      const x = (index / (waveformData.length - 1)) * sliderWidth + 20;
      
      // Apply privacy distortion to waveform
      let morphedAmplitude = amplitude;
      
      if (privacyLevel > 0.3) {
        // Add quantization effect for higher privacy
        const quantizationLevels = Math.floor((1 - privacyLevel) * 10) + 2;
        morphedAmplitude = Math.round(amplitude * quantizationLevels) / quantizationLevels;
      }
      
      if (privacyLevel > 0.7) {
        // Add noise and smoothing for very high privacy
        morphedAmplitude += (Math.random() - 0.5) * privacyLevel * 0.3;
        morphedAmplitude *= (1 - privacyLevel * 0.4); // Reduce amplitude
      }
      
      const y = baselineY + (morphedAmplitude * height * 0.3);
      points.push(`${index === 0 ? 'M' : 'L'} ${x} ${y}`);
    });
    
    return points.join(' ');
  };

  // Animated styles for the slider track
  const trackStyle = useAnimatedStyle(() => {
    const currentValue = thumbPosition.value / sliderWidth;
    
    return {
      backgroundColor: interpolateColor(
        currentValue,
        [0, 0.5, 1],
        ['#ff4444', '#ffaa00', '#00ff88']
      ),
    };
  });

  // Animated styles for the thumb
  const thumbStyle = useAnimatedStyle(() => {
    const currentValue = thumbPosition.value / sliderWidth;
    
    return {
      transform: [
        { translateX: thumbPosition.value },
        { scale: isDragging.value ? withSpring(1.2) : withSpring(1) }
      ],
      backgroundColor: interpolateColor(
        currentValue,
        [0, 0.5, 1],
        ['#ff4444', '#ffaa00', '#00ff88']
      ),
      shadowColor: interpolateColor(
        currentValue,
        [0, 0.5, 1],
        ['#ff4444', '#ffaa00', '#00ff88']
      ),
    };
  });

  // Animated waveform morphing
  const waveformStyle = useAnimatedStyle(() => {
    const currentValue = thumbPosition.value / sliderWidth;
    
    return {
      opacity: interpolate(currentValue, [0, 1], [0.6, 1]),
      transform: [{
        scaleY: interpolate(currentValue, [0, 1], [1, 0.7])
      }]
    };
  });

  return (
    <View style={[styles.container, style, { width, height }]}>
      {/* Waveform Visualization Background */}
      {visualFeedback === 'waveform-morphing' && (
        <Animated.View style={[styles.waveformContainer, waveformStyle]}>
          <Svg width={width} height={height}>
            <Defs>
              <LinearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <Stop offset="0%" stopColor="#ff4444" stopOpacity="0.3" />
                <Stop offset="50%" stopColor="#ffaa00" stopOpacity="0.3" />
                <Stop offset="100%" stopColor="#00ff88" stopOpacity="0.3" />
              </LinearGradient>
            </Defs>
            
            <Path
              d={generateWaveformPath(thumbPosition.value / sliderWidth)}
              stroke="url(#waveGradient)"
              strokeWidth="2"
              fill="none"
              strokeLinecap="round"
            />
          </Svg>
        </Animated.View>
      )}

      {/* Slider Track */}
      <Animated.View style={[styles.track, trackStyle]} />

      {/* Privacy Level Indicators */}
      <View style={styles.indicators}>
        <View style={[styles.indicator, { left: '10%' }]}>
          <View style={[styles.indicatorDot, { backgroundColor: '#ff4444' }]} />
          <Text style={styles.indicatorLabel}>Low</Text>
        </View>
        <View style={[styles.indicator, { left: '50%', marginLeft: -15 }]}>
          <View style={[styles.indicatorDot, { backgroundColor: '#ffaa00' }]} />
          <Text style={styles.indicatorLabel}>Med</Text>
        </View>
        <View style={[styles.indicator, { right: '10%' }]}>
          <View style={[styles.indicatorDot, { backgroundColor: '#00ff88' }]} />
          <Text style={styles.indicatorLabel}>High</Text>
        </View>
      </View>

      {/* Slider Thumb */}
      <PanGestureHandler onGestureEvent={panGestureHandler}>
        <Animated.View style={[styles.thumb, thumbStyle]}>
          <LynxAnimations.Pulse duration={1500}>
            <View style={styles.thumbInner}>
              <Text style={styles.thumbIcon}>🛡️</Text>
            </View>
          </LynxAnimations.Pulse>
          
          {/* Privacy level ring */}
          <Animated.View style={[styles.thumbRing, thumbStyle]} />
        </Animated.View>
      </PanGestureHandler>

      {/* Privacy Impact Visualization */}
      <View style={styles.impactContainer}>
        <Svg width={width} height="30">
          {/* Voice characteristics bars */}
          {Array.from({ length: 8 }, (_, i) => {
            const x = (i / 7) * (width - 40) + 20;
            const currentValue = thumbPosition.value / sliderWidth;
            const barHeight = interpolate(
              currentValue,
              [0, 1],
              [25, 10 - (i * 1.5)] // Higher privacy = more distortion
            );
            
            return (
              <Animated.View key={i}>
                <Circle
                  cx={x}
                  cy={15}
                  r={barHeight / 2}
                  fill={interpolateColor(
                    currentValue,
                    [0, 0.5, 1],
                    ['rgba(255, 68, 68, 0.6)', 'rgba(255, 170, 0, 0.6)', 'rgba(0, 255, 136, 0.6)']
                  )}
                />
              </Animated.View>
            );
          })}
        </Svg>
      </View>

      {/* Privacy Protection Labels */}
      <View style={styles.protectionLabels}>
        <Text style={styles.protectionLabel}>Voice Biometrics</Text>
        <Text style={styles.protectionLabel}>Emotional Patterns</Text>
        <Text style={styles.protectionLabel}>Background Voices</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: 15,
    padding: 20,
    position: 'relative',
  },
  waveformContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderRadius: 15,
  },
  track: {
    height: 8,
    borderRadius: 4,
    marginHorizontal: 20,
    marginTop: 20,
    opacity: 0.3,
  },
  thumb: {
    position: 'absolute',
    top: 16,
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    marginLeft: 20,
  },
  thumbInner: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  thumbIcon: {
    fontSize: 12,
  },
  thumbRing: {
    position: 'absolute',
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  indicators: {
    flexDirection: 'row',
    position: 'absolute',
    top: 50,
    left: 0,
    right: 0,
  },
  indicator: {
    alignItems: 'center',
    position: 'absolute',
  },
  indicatorDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginBottom: 4,
  },
  indicatorLabel: {
    color: '#ffffff',
    fontSize: 10,
    opacity: 0.7,
  },
  impactContainer: {
    marginTop: 20,
    height: 30,
  },
  protectionLabels: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 10,
  },
  protectionLabel: {
    color: '#ffffff',
    fontSize: 10,
    opacity: 0.6,
    textAlign: 'center',
  },
});

export default PrivacySlider;
