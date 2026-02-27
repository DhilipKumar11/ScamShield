import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as DocumentPicker from 'expo-document-picker';
import { analyzeAudio } from '../../src/services/api';

// ─── Helpers ────────────────────────────────────────────────────────────────

const riskColor = (score: number) => {
  if (score >= 70) return '#ef4444';
  if (score >= 40) return '#f59e0b';
  return '#22c55e';
};

const riskLabel = (score: number) => {
  if (score >= 70) return 'HIGH RISK';
  if (score >= 40) return 'MEDIUM RISK';
  return 'SAFE';
};

const verdictColor = (verdict: string) => {
  if (verdict === 'AI_VOICE') return '#ef4444';
  if (verdict === 'HUMAN_VOICE') return '#22c55e';
  return '#f59e0b';
};

const verdictIcon = (verdict: string) => {
  if (verdict === 'AI_VOICE') return '🤖';
  if (verdict === 'HUMAN_VOICE') return '👤';
  return '❓';
};

const verdictLabel = (verdict: string) => {
  if (verdict === 'AI_VOICE') return 'AI Generated Voice';
  if (verdict === 'HUMAN_VOICE') return 'Human Voice';
  return 'Inconclusive';
};

// ─── Sub-components ─────────────────────────────────────────────────────────

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>{title}</Text>
      {children}
    </View>
  );
}

function MetricRow({
  label,
  value,
  valueColor,
  bar,
  barColor,
}: {
  label: string;
  value: string;
  valueColor?: string;
  bar?: number;
  barColor?: string;
}) {
  return (
    <View style={styles.metricRow}>
      <View style={styles.metricTopRow}>
        <Text style={styles.metricLabel}>{label}</Text>
        <Text style={[styles.metricValue, valueColor ? { color: valueColor } : {}]}>{value}</Text>
      </View>
      {bar !== undefined && (
        <View style={styles.barTrack}>
          <View
            style={[
              styles.barFill,
              { width: `${Math.min(bar, 100)}%` as any, backgroundColor: barColor || '#0ea5e9' },
            ]}
          />
        </View>
      )}
    </View>
  );
}

function RiskGauge({ score }: { score: number }) {
  const color = riskColor(score);
  const label = riskLabel(score);
  return (
    <View style={[styles.riskGauge, { borderColor: color }]}>
      <Text style={[styles.riskScore, { color }]}>{score}</Text>
      <Text style={styles.riskScoreMax}>/100</Text>
      <View style={[styles.riskBadge, { backgroundColor: color + '22' }]}>
        <Text style={[styles.riskBadgeText, { color }]}>{label}</Text>
      </View>
    </View>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function AudioUploadMVP() {
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState('');

  const handleFileUpload = async () => {
    try {
      const pickerResult = await DocumentPicker.getDocumentAsync({
        type: 'audio/*',
        copyToCacheDirectory: true,
      });

      if (pickerResult.canceled) return;

      const file = pickerResult.assets[0];
      setLoading(true);
      setErrorMsg('');
      setResultData(null);

      const analysisResult = await analyzeAudio(file.uri, file.name, file.mimeType);
      setResultData(analysisResult);
    } catch (error) {
      console.error(error);
      setErrorMsg('Upload failed. Ensure the Python backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>

        {/* ── Header ── */}
        <View style={styles.header}>
          <Text style={styles.title}>🛡️ ScamShield</Text>
          <Text style={styles.subtitle}>Audio Scam & Voice Analysis</Text>
        </View>

        {/* ── Upload Button ── */}
        <TouchableOpacity
          style={[styles.uploadButton, loading && styles.buttonDisabled]}
          onPress={handleFileUpload}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#ffffff" />
          ) : (
            <Text style={styles.uploadButtonText}>
              {resultData ? '🔄 Analyze Another File' : '📂 Upload Audio File'}
            </Text>
          )}
        </TouchableOpacity>

        {loading && (
          <View style={styles.loadingBox}>
            <ActivityIndicator color="#0ea5e9" size="large" />
            <Text style={styles.statusText}>Analyzing audio — transcribing & running voice AI...</Text>
          </View>
        )}

        {errorMsg !== '' && <Text style={styles.errorText}>{errorMsg}</Text>}

        {/* ── Results ── */}
        {resultData && (
          <View style={styles.resultsWrapper}>

            {/* 1. Risk Score */}
            <SectionCard title="⚠️  Overall Risk Score">
              <RiskGauge score={Math.round(resultData.risk_score)} />
              <Text style={styles.insightText}>{resultData.insight}</Text>
            </SectionCard>

            {/* 2. Voice Verdict */}
            <SectionCard title="🎙️  Voice Verdict">
              <View style={[styles.verdictBanner, { backgroundColor: verdictColor(resultData.verdict) + '18' }]}>
                <Text style={styles.verdictIcon}>{verdictIcon(resultData.verdict)}</Text>
                <View>
                  <Text style={[styles.verdictLabel, { color: verdictColor(resultData.verdict) }]}>
                    {verdictLabel(resultData.verdict)}
                  </Text>
                  <Text style={styles.verdictConfidence}>
                    Confidence: {Math.round(resultData.verdict_confidence)}%
                  </Text>
                </View>
              </View>
              <MetricRow
                label="Synthetic Probability"
                value={`${Math.round(resultData.voice_score)}%`}
                valueColor={resultData.voice_score > 50 ? '#ef4444' : '#22c55e'}
                bar={resultData.voice_score}
                barColor={resultData.voice_score > 50 ? '#ef4444' : '#22c55e'}
              />
              <MetricRow
                label="ML Score"
                value={`${resultData.ml_score?.toFixed(1)}%`}
                bar={resultData.ml_score}
                barColor="#8b5cf6"
              />
              <MetricRow
                label="KNN Score"
                value={`${resultData.knn_score?.toFixed(1)}%`}
                bar={resultData.knn_score}
                barColor="#06b6d4"
              />
            </SectionCard>

            {/* 3. Content Analysis */}
            <SectionCard title="📊  Content Analysis">
              <MetricRow
                label="Content Risk Score"
                value={`${Math.round(resultData.content_score)}/100`}
                valueColor={riskColor(resultData.content_score)}
                bar={resultData.content_score}
                barColor={riskColor(resultData.content_score)}
              />
              <MetricRow
                label="Confidence"
                value={`${resultData.confidence}%`}
                bar={resultData.confidence}
                barColor="#0ea5e9"
              />
              <View style={styles.infoRow}>
                <Text style={styles.metricLabel}>Threat Category</Text>
                <Text style={[styles.metricValue, { color: resultData.threat_category !== 'None' ? '#ef4444' : '#22c55e' }]}>
                  {resultData.threat_category || 'None'}
                </Text>
              </View>
              {resultData.categories?.length > 0 && (
                <View style={styles.tagsRow}>
                  {resultData.categories.map((cat: string, i: number) => (
                    <View key={i} style={styles.tag}>
                      <Text style={styles.tagText}>{cat}</Text>
                    </View>
                  ))}
                </View>
              )}
              {resultData.matched_keywords?.length > 0 && (
                <View style={{ marginTop: 8 }}>
                  <Text style={styles.metricLabel}>Flagged Keywords</Text>
                  <Text style={styles.keywordsText}>{resultData.matched_keywords.join(', ')}</Text>
                </View>
              )}
            </SectionCard>

            {/* 4. Acoustic Details */}
            {resultData.acoustic_details && (
              <SectionCard title="🔬  Acoustic Details">
                <View style={styles.acousticGrid}>
                  <AcousticCell label="Pitch Mean" value={`${resultData.acoustic_details.pitch_mean_hz} Hz`} />
                  <AcousticCell label="Pitch Std" value={`${resultData.acoustic_details.pitch_std_hz} Hz`} />
                  <AcousticCell label="Jitter" value={`${resultData.acoustic_details.jitter_pct?.toFixed(2)}%`} />
                  <AcousticCell label="Shimmer" value={`${resultData.acoustic_details.shimmer_pct?.toFixed(1)}%`} />
                  <AcousticCell label="HNR" value={`${resultData.acoustic_details.hnr_db?.toFixed(1)} dB`} />
                  <AcousticCell label="Voiced Ratio" value={`${resultData.acoustic_details.voiced_ratio_pct?.toFixed(1)}%`} />
                  <AcousticCell label="Spectral Flatness" value={resultData.acoustic_details.spectral_flatness?.toFixed(4)} />
                  <AcousticCell label="Duration" value={`${resultData.acoustic_details.duration_s?.toFixed(2)} s`} />
                </View>
              </SectionCard>
            )}

            {/* 5. Voice Panel Data */}
            {resultData.voice_data && (
              <SectionCard title="📈  Voice Biomarkers">
                <MetricRow
                  label="Naturalness"
                  value={`${resultData.voice_data.naturalness_score?.toFixed(1)}%`}
                  bar={resultData.voice_data.naturalness_score}
                  barColor="#22c55e"
                />
                <MetricRow
                  label="Pitch Variance"
                  value={`${resultData.voice_data.pitch_variance}%`}
                  bar={resultData.voice_data.pitch_variance}
                  barColor="#8b5cf6"
                />
                <MetricRow
                  label="MFCC Stability"
                  value={`${resultData.voice_data.mfcc_stability?.toFixed(1)}%`}
                  bar={resultData.voice_data.mfcc_stability}
                  barColor="#0ea5e9"
                />
                <MetricRow
                  label="Spectral Smoothness"
                  value={`${resultData.voice_data.spectral_smoothness?.toFixed(1)}%`}
                  bar={resultData.voice_data.spectral_smoothness}
                  barColor="#f59e0b"
                />
              </SectionCard>
            )}

            {/* 6. KNN Nearest Matches */}
            {resultData.nearest_matches?.length > 0 && (
              <SectionCard title="🔍  Nearest Voice Matches (KNN)">
                {resultData.nearest_matches.map((m: any) => (
                  <View key={m.rank} style={styles.matchRow}>
                    <View style={styles.matchRank}>
                      <Text style={styles.matchRankText}>#{m.rank}</Text>
                    </View>
                    <View style={styles.matchInfo}>
                      <Text style={styles.matchFile} numberOfLines={1}>{m.file}</Text>
                      <View style={styles.matchMeta}>
                        <Text style={[
                          styles.matchLabel,
                          { color: m.label === 'AI' ? '#ef4444' : '#22c55e' }
                        ]}>
                          {m.label === 'AI' ? '🤖 AI Voice' : '👤 Human'}
                        </Text>
                        <Text style={styles.matchSimilarity}>{m.similarity}% match</Text>
                      </View>
                    </View>
                    <View style={styles.matchBarWrap}>
                      <View style={[styles.matchBar, { width: `${m.similarity}%` as any, backgroundColor: m.label === 'AI' ? '#ef444466' : '#22c55e66' }]} />
                    </View>
                  </View>
                ))}
              </SectionCard>
            )}

            {/* 7. Transcript */}
            {resultData.transcript_entries?.length > 0 && (
              <SectionCard title="📝  Transcript">
                {resultData.transcript_entries.map((entry: any, i: number) => (
                  <View key={i} style={[styles.transcriptEntry, entry.flagged && styles.transcriptFlagged]}>
                    <View style={styles.transcriptMeta}>
                      <Text style={styles.transcriptTime}>{entry.time}</Text>
                      <Text style={styles.transcriptSpeaker}>{entry.speaker}</Text>
                      {entry.flagged && <View style={styles.flagDot} />}
                    </View>
                    <Text style={styles.transcriptText}>{entry.text}</Text>
                    {entry.flagged && entry.categories?.length > 0 && (
                      <Text style={styles.transcriptCategories}>⚠️ {entry.categories.join(', ')}</Text>
                    )}
                  </View>
                ))}
                <View style={styles.languageRow}>
                  <Text style={styles.metricLabel}>Detected Language</Text>
                  <Text style={styles.metricValue}>{resultData.language?.toUpperCase()}</Text>
                </View>
              </SectionCard>
            )}

          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function AcousticCell({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.acousticCell}>
      <Text style={styles.acousticValue}>{value}</Text>
      <Text style={styles.acousticLabel}>{label}</Text>
    </View>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0f1e' },
  scrollContent: { padding: 20, paddingBottom: 60 },

  // Header
  header: { marginBottom: 28, alignItems: 'center' },
  title: { fontSize: 30, fontWeight: '800', color: '#f0f6ff', letterSpacing: 0.5 },
  subtitle: { fontSize: 14, color: '#64748b', marginTop: 4 },

  // Upload
  uploadButton: {
    backgroundColor: '#0ea5e9',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#0ea5e9',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 6,
  },
  buttonDisabled: { backgroundColor: '#1e293b' },
  uploadButtonText: { color: '#fff', fontSize: 16, fontWeight: '700' },

  // Loading
  loadingBox: { alignItems: 'center', paddingVertical: 24, gap: 12 },
  statusText: { color: '#94a3b8', textAlign: 'center', fontSize: 14 },
  errorText: { color: '#ef4444', textAlign: 'center', marginBottom: 16, fontWeight: '600' },

  // Card
  resultsWrapper: { gap: 16 },
  card: {
    backgroundColor: '#111827',
    borderRadius: 16,
    padding: 18,
    borderWidth: 1,
    borderColor: '#1e293b',
  },
  cardTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#64748b',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
    marginBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#1e293b',
    paddingBottom: 10,
  },

  // Risk Gauge
  riskGauge: {
    alignItems: 'center',
    borderRadius: 100,
    borderWidth: 4,
    width: 130,
    height: 130,
    alignSelf: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  riskScore: { fontSize: 46, fontWeight: '900', lineHeight: 50 },
  riskScoreMax: { fontSize: 13, color: '#475569', fontWeight: '600' },
  riskBadge: { paddingHorizontal: 12, paddingVertical: 4, borderRadius: 99, marginTop: 6 },
  riskBadgeText: { fontSize: 11, fontWeight: '800', letterSpacing: 1 },
  insightText: { color: '#94a3b8', fontSize: 13, lineHeight: 20, fontStyle: 'italic', textAlign: 'center' },

  // Verdict
  verdictBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    borderRadius: 12,
    marginBottom: 16,
    gap: 12,
  },
  verdictIcon: { fontSize: 32 },
  verdictLabel: { fontSize: 17, fontWeight: '800' },
  verdictConfidence: { color: '#64748b', fontSize: 12, marginTop: 2 },

  // Metric
  metricRow: { marginBottom: 14 },
  metricTopRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 },
  metricLabel: { color: '#64748b', fontSize: 13, fontWeight: '600' },
  metricValue: { color: '#e2e8f0', fontSize: 13, fontWeight: '700' },
  barTrack: { height: 6, backgroundColor: '#1e293b', borderRadius: 99, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 99 },
  infoRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10 },

  // Tags
  tagsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 8 },
  tag: { backgroundColor: '#ef444422', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 99 },
  tagText: { color: '#ef4444', fontSize: 11, fontWeight: '700' },
  keywordsText: { color: '#f59e0b', fontSize: 12, marginTop: 4, fontStyle: 'italic' },

  // Acoustic Grid
  acousticGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  acousticCell: {
    backgroundColor: '#0f172a',
    borderRadius: 10,
    padding: 12,
    width: '47%',
    alignItems: 'center',
  },
  acousticValue: { color: '#e2e8f0', fontSize: 16, fontWeight: '700' },
  acousticLabel: { color: '#475569', fontSize: 11, marginTop: 4, textAlign: 'center' },

  // KNN Matches
  matchRow: { marginBottom: 12 },
  matchRank: {
    position: 'absolute',
    left: 0,
    top: 0,
    width: 28,
    height: 28,
    borderRadius: 99,
    backgroundColor: '#1e293b',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
  },
  matchRankText: { color: '#94a3b8', fontSize: 11, fontWeight: '800' },
  matchInfo: { marginLeft: 36 },
  matchFile: { color: '#cbd5e1', fontSize: 13, fontWeight: '600' },
  matchMeta: { flexDirection: 'row', gap: 12, marginTop: 3 },
  matchLabel: { fontSize: 12, fontWeight: '700' },
  matchSimilarity: { color: '#64748b', fontSize: 12 },
  matchBarWrap: {
    height: 4,
    backgroundColor: '#1e293b',
    borderRadius: 99,
    overflow: 'hidden',
    marginTop: 6,
    marginLeft: 36,
  },
  matchBar: { height: '100%', borderRadius: 99 },

  // Transcript
  transcriptEntry: {
    backgroundColor: '#0f172a',
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#1e293b',
  },
  transcriptFlagged: { borderLeftColor: '#ef4444' },
  transcriptMeta: { flexDirection: 'row', gap: 10, alignItems: 'center', marginBottom: 6 },
  transcriptTime: { color: '#475569', fontSize: 11, fontWeight: '600' },
  transcriptSpeaker: {
    color: '#0ea5e9',
    fontSize: 11,
    fontWeight: '700',
    backgroundColor: '#0ea5e922',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  flagDot: { width: 7, height: 7, borderRadius: 99, backgroundColor: '#ef4444', marginLeft: 'auto' as any },
  transcriptText: { color: '#cbd5e1', fontSize: 13, lineHeight: 20 },
  transcriptCategories: { color: '#f59e0b', fontSize: 11, marginTop: 6 },
  languageRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 12,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#1e293b',
  },
});
