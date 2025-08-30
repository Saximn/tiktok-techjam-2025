/**
 * VoiceSpectrogram - Real-time audio visualization with privacy masking
 * Shows voice frequency patterns while protecting biometric signatures
 */

import React, { useEffect, useRef } from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Path, Rect, Defs, LinearGradient, Stop, Mask } from 'react-native-svg';
import { LynxAnimations } from '@lynx-ui/core';
import Animated, { 
  useSharedValue, 
  useAnimatedStyle, 
  withSpring,
  interpolate,
  runOnJS
} from 'react-native-reanimated';

export const VoiceSpectrogram = ({
  realtime = false,
  privacyMask = false,
  amplitude = 0,
  frequency = 440,
  style,
  width = 300,
  height = 200
}) => {
  const canvasRef = useRef(null);
  const animatedAmplitude = useSharedValue(0);
  const animatedFrequency = useSharedValue(440);
  const privacyMaskOpacity = useSharedValue(privacyMask ? 0.3 : 0);
  
  // Store frequency data for spectrogram display
  const [frequencyData, setFrequencyData] = React.useState(
    Array.from({ length: 64 }, () => Array.from({ length: 100 }, () => Math.random() * 0.1))
  );

  // Update animations when props change
  useEffect(() => {
    animatedAmplitude.value = withSpring(amplitude, { damping: 10, stiffness: 100 });
    animatedFrequency.value = withSpring(frequency, { damping: 15, stiffness: 120 });
    privacyMaskOpacity.value = withSpring(privacyMask ? 0.4 : 0);
  }, [amplitude, frequency, privacyMask]);

  // Generate realistic spectrogram data
  const generateSpectrogramData = () => {
    const newData = frequencyData.map((freqBin, freqIndex) => {
      const newBin = [...freqBin];
      newBin.shift(); // Remove oldest sample
      
      // Generate new sample based on current audio
      let newSample = 0;
      
      if (amplitude > 0.01) {
        // Voice activity detected
        const normalizedFreq = frequency / 4000; // Normalize to 0-1
        const freqBinCenter = freqIndex / 64; // Current frequency bin center
        
        // Create realistic voice formant patterns
        const formant1 = Math.exp(-Math.pow((freqBinCenter - 0.2) * 10, 2)) * amplitude * 0.8;
        const formant2 = Math.exp(-Math.pow((freqBinCenter - 0.5) * 8, 2)) * amplitude * 0.6;
        const formant3 = Math.exp(-Math.pow((freqBinCenter - 0.8) * 6, 2)) * amplitude * 0.4;
        
        newSample = formant1 + formant2 + formant3 + (Math.random() * 0.1);
        
        // Apply privacy masking - reduce formant clarity
        if (privacyMask) {
          newSample = newSample * 0.7 + (Math.random() * 0.2);
        }
      } else {
        // Background noise
        newSample = Math.random() * 0.05;
      }
      
      newBin.push(Math.max(0, Math.min(1, newSample)));
      return newBin;
    });
    
    setFrequencyData(newData);
  };

  // Update spectrogram data in real-time
  useEffect(() => {
    if (!realtime) return;
    
    const interval = setInterval(generateSpectrogramData, 50); // 20fps
    return () => clearInterval(interval);
  }, [realtime, amplitude, frequency, privacyMask]);

  // Generate SVG path for waveform overlay
  const generateWaveformPath = () => {
    const points = [];
    const samples = 100;
    
    for (let i = 0; i < samples; i++) {
      const x = (i / samples) * width;
      const baseY = height * 0.5;
      
      // Create wave pattern based on current amplitude and frequency
      const time = i / samples * 4 * Math.PI;
      const wave = Math.sin(time * (frequency / 440)) * amplitude * height * 0.3;
      
      // Add privacy distortion if enabled
      let y = baseY + wave;
      if (privacyMask) {
        const distortion = Math.sin(time * 3.7) * amplitude * height * 0.1;
        y += distortion;
      }
      
      points.push(`${i === 0 ? 'M' : 'L'} ${x} ${y}`);
    }
    
    return points.join(' ');
  };

  // Animated styles for privacy mask
  const privacyMaskStyle = useAnimatedStyle(() => ({
    opacity: privacyMaskOpacity.value,
  }));

  const waveformStyle = useAnimatedStyle(() => ({
    opacity: interpolate(animatedAmplitude.value, [0, 1], [0.3, 1]),
    transform: [{
      scaleY: interpolate(animatedAmplitude.value, [0, 1], [0.5, 1.2])
    }]
  }));

  return (
    <View style={[styles.container, style, { width, height }]}>
      {/* Spectrogram Background */}
      <View style={styles.spectrogramContainer}>
        {frequencyData.map((freqBin, freqIndex) => (
          <View key={freqIndex} style={styles.frequencyBin}>
            {freqBin.map((intensity, timeIndex) => (
              <View
                key={timeIndex}
                style={[
                  styles.spectrogramPixel,
                  {
                    opacity: intensity,
                    backgroundColor: getFrequencyColor(freqIndex / 64, intensity)
                  }
                ]}
              />
            ))}
          </View>
        ))}
      </View>

      {/* Real-time Waveform Overlay */}
      <Animated.View style={[StyleSheet.absoluteFill, waveformStyle]}>
        <Svg width={width} height={height}>
          <Defs>
            <LinearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <Stop offset="0%" stopColor="#00ff88" stopOpacity="0.8" />
              <Stop offset="50%" stopColor="#ffaa00" stopOpacity="0.9" />
              <Stop offset="100%" stopColor="#ff4444" stopOpacity="0.8" />
            </LinearGradient>
          </Defs>
          
          <Path
            d={generateWaveformPath()}
            stroke="url(#waveGradient)"
            strokeWidth="2"
            fill="none"
            strokeLinecap="round"
          />
        </Svg>
      </Animated.View>

      {/* Privacy Mask Overlay */}
      {privacyMask && (
        <Animated.View style={[styles.privacyOverlay, privacyMaskStyle]}>
          <Svg width={width} height={height}>
            <Defs>
              <Mask id="privacyMask">
                <Rect width={width} height={height} fill="white" />
                {/* Create masking pattern */}
                {Array.from({ length: 20 }, (_, i) => (
                  <Rect
                    key={i}
                    x={Math.random() * width}
                    y={Math.random() * height}
                    width="30"
                    height="20"
                    fill="black"
                    opacity="0.6"
                  />
                ))}
              </Mask>
            </Defs>
            
            <Rect
              width={width}
              height={height}
              fill="#4400ff"
              mask="url(#privacyMask)"
              opacity="0.5"
            />
          </Svg>
        </Animated.View>
      )}

      {/* Privacy Status Indicator */}
      {privacyMask && (
        <View style={styles.privacyIndicator}>
          <LynxAnimations.Pulse duration={1000}>
            <View style={styles.privacyIcon}>
              <Text style={styles.privacyIconText}>🛡️</Text>
            </View>
          </LynxAnimations.Pulse>
        </View>
      )}
    </View>
  );
};

// Helper function to get frequency-based colors
const getFrequencyColor = (normalizedFreq, intensity) => {
  if (intensity < 0.1) return `rgba(20, 20, 40, ${intensity})`;
  
  const hue = normalizedFreq * 240; // 0° (red) to 240° (blue)
  const saturation = 70 + (intensity * 30); // 70% to 100%
  const lightness = 30 + (intensity * 40); // 30% to 70%
  
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#0a0a0a',
    borderRadius: 10,
    overflow: 'hidden',
    position: 'relative',
  },
  spectrogramContainer: {
    flex: 1,
    flexDirection: 'row',
  },
  frequencyBin: {
    flex: 1,
    flexDirection: 'column-reverse', // Low frequencies at bottom
  },
  spectrogramPixel: {
    flex: 1,
    minHeight: 2,
  },
  privacyOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(68, 0, 255, 0.2)',
  },
  privacyIndicator: {
    position: 'absolute',
    top: 10,
    right: 10,
  },
  privacyIcon: {
    backgroundColor: 'rgba(0, 255, 136, 0.2)',
    borderRadius: 15,
    width: 30,
    height: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  privacyIconText: {
    fontSize: 16,
  },
});

export default VoiceSpectrogram;
