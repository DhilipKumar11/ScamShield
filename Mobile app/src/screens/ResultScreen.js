import React from 'react';
import { StyleSheet, Text, View, ScrollView, TouchableOpacity } from 'react-native';

export default function ResultScreen({ route, navigation }) {
  // Get the data passed from HomeScreen
  const { data } = route.params;

  // Determine styles based on risk score (Assuming 0-100 score, higher is worse)
  const isHighRisk = data?.riskScore > 60;
  const isMediumRisk = data?.riskScore > 30 && data?.riskScore <= 60;

  const headerColor = isHighRisk ? '#ef4444' : isMediumRisk ? '#eab308' : '#22c55e';
  const statusText = isHighRisk ? 'High Risk Scam' : isMediumRisk ? 'Suspicious' : 'Appears Safe';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      
      <View style={[styles.statusCard, { borderColor: headerColor }]}>
        <View style={[styles.statusIndicator, { backgroundColor: headerColor }]} />
        <Text style={styles.statusTitle}>{statusText}</Text>
        <Text style={styles.scoreText}>Risk Score: {data?.riskScore || 0}/100</Text>
      </View>

      <View style={styles.detailsCard}>
        <Text style={styles.sectionTitle}>Analysis Details</Text>
        
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Type:</Text>
          <Text style={styles.detailValue}>{data?.type || 'Unknown'}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Confidence:</Text>
          <Text style={styles.detailValue}>{data?.confidence || '0'}%</Text>
        </View>

        {data?.reasons && data.reasons.length > 0 && (
          <View style={styles.reasonsContainer}>
            <Text style={styles.detailLabel}>Flags Detected:</Text>
            {data.reasons.map((reason, index) => (
              <Text key={index} style={styles.reasonItem}>• {reason}</Text>
            ))}
          </View>
        )}
      </View>

      {/* Voice Specific Data (Only shows if an audio file was uploaded) */}
      {data?.transcript && (
        <View style={styles.detailsCard}>
          <Text style={styles.sectionTitle}>Audio Analysis</Text>

          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Voice Detection:</Text>
            <Text style={[
              styles.detailValue, 
              { color: data.verdict === 'AI_VOICE' ? '#ef4444' : '#22c55e' }
            ]}>
              {data.verdict === 'AI_VOICE' ? 'AI Generated' : data.verdict === 'HUMAN_VOICE' ? 'Human Voice' : 'Unknown'}
            </Text>
          </View>

          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>AI Voice Probability:</Text>
            <Text style={styles.detailValue}>{data.voice_score || 0}%</Text>
          </View>

          <View style={styles.reasonsContainer}>
            <Text style={styles.detailLabel}>Transcript:</Text>
            <Text style={styles.transcriptText}>"{data.transcript}"</Text>
          </View>
        </View>
      )}

      <TouchableOpacity 
        style={styles.backButton} 
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.backButtonText}>Scan Another</Text>
      </TouchableOpacity>
      
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  content: {
    padding: 24,
  },
  statusCard: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    borderWidth: 2,
    marginBottom: 24,
  },
  statusIndicator: {
    width: 64,
    height: 64,
    borderRadius: 32,
    marginBottom: 16,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 10,
  },
  statusTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#f8fafc',
    marginBottom: 8,
  },
  scoreText: {
    fontSize: 18,
    color: '#cbd5e1',
    fontWeight: '600',
  },
  detailsCard: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 24,
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#f8fafc',
    marginBottom: 24,
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
    paddingBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  detailLabel: {
    fontSize: 16,
    color: '#94a3b8',
    fontWeight: '600',
  },
  detailValue: {
    fontSize: 16,
    color: '#f8fafc',
    fontWeight: 'bold',
  },
  reasonsContainer: {
    marginTop: 8,
  },
  reasonItem: {
    color: '#ef4444',
    fontSize: 15,
    marginTop: 8,
    lineHeight: 22,
    marginLeft: 8,
  },
  transcriptText: {
    color: '#cbd5e1',
    fontSize: 15,
    marginTop: 12,
    lineHeight: 24,
    fontStyle: 'italic',
    backgroundColor: '#0f172a',
    padding: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#334155'
  },
  backButton: {
    backgroundColor: '#334155',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  backButtonText: {
    color: '#f8fafc',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
