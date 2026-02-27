import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TextInput, 
  TouchableOpacity, 
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { analyzeContent, analyzeAudio } from '../services/api';

export default function HomeScreen({ navigation }) {
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      Alert.alert('Error', 'Please enter a URL or text message to analyze.', [{ text: 'OK' }]);
      return;
    }

    setLoading(true);
    setLoadingMessage('Analyzing text...');
    try {
      const result = await analyzeContent(inputText);
      navigation.navigate('Result', { data: result });
    } catch (error) {
      console.error(error);
      Alert.alert('Analysis Failed', 'Could not connect to the backend.', [{ text: 'OK' }]);
    } finally {
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const handleFileUpload = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'audio/*', // Restrict to audio files
        copyToCacheDirectory: true,
      });

      if (result.canceled) {
        return; // User cancelled the picker
      }

      const file = result.assets[0];
      
      setLoading(true);
      setLoadingMessage('Uploading & Analyzing Audio...');
      
      // Step 2: Send file to backend
      const analysisResult = await analyzeAudio(file.uri, file.name, file.mimeType);
      
      // Step 3: Navigate to Result screen with the response data
      navigation.navigate('Result', { data: analysisResult });

    } catch (error) {
      console.error(error);
      Alert.alert('Upload Failed', 'There was a problem uploading the audio file. Make sure the backend is running.', [{ text: 'OK' }]);
    } finally {
      setLoading(false);
      setLoadingMessage('');
    }
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        
        <View style={styles.header}>
          <Text style={styles.title}>Welcome back.</Text>
          <Text style={styles.subtitle}>Protect yourself from scams. Paste any suspicious link, SMS, or message below.</Text>
        </View>

        <View style={styles.inputContainer}>
          <Text style={styles.label}>Content to Analyze</Text>
          <TextInput
            style={styles.textInput}
            multiline
            placeholder="e.g. You have won $10,000! Click here: http://..."
            placeholderTextColor="#475569"
            value={inputText}
            onChangeText={setInputText}
          />
        </View>

        <TouchableOpacity 
          style={[styles.button, loading && styles.buttonDisabled]} 
          onPress={handleAnalyze}
          disabled={loading}
        >
          {loading && loadingMessage === 'Analyzing text...' ? (
            <ActivityIndicator color="#0f172a" />
          ) : (
            <Text style={styles.buttonText}>Scan Text / URL</Text>
          )}
        </TouchableOpacity>

        <View style={styles.divider}>
          <View style={styles.line} />
          <Text style={styles.orText}>OR</Text>
          <View style={styles.line} />
        </View>

        <TouchableOpacity 
          style={[styles.uploadButton, loading && styles.buttonDisabled]} 
          onPress={handleFileUpload}
          disabled={loading}
        >
          {loading && loadingMessage.includes('Audio') ? (
            <ActivityIndicator color="#f8fafc" />
          ) : (
            <Text style={styles.uploadButtonText}>Upload Audio (Call Recording)</Text>
          )}
        </TouchableOpacity>

        {loading && <Text style={styles.loadingMessage}>{loadingMessage}</Text>}

      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  scrollContent: {
    flexGrow: 1,
    padding: 24,
    justifyContent: 'center',
  },
  header: {
    marginBottom: 40,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#f8fafc',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#94a3b8',
    lineHeight: 24,
  },
  inputContainer: {
    marginBottom: 32,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#e2e8f0',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  textInput: {
    backgroundColor: '#1e293b',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#334155',
    color: '#f8fafc',
    padding: 16,
    minHeight: 150,
    textAlignVertical: 'top',
    fontSize: 16,
  },
  button: {
    backgroundColor: '#38bdf8',
    borderRadius: 12,
    paddingVertical: 18,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#38bdf8',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  buttonDisabled: {
    backgroundColor: '#0284c7',
    opacity: 0.7,
  },
  buttonText: {
    color: '#0f172a',
    fontSize: 18,
    fontWeight: 'bold',
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 24,
  },
  line: {
    flex: 1,
    height: 1,
    backgroundColor: '#334155',
  },
  orText: {
    color: '#94a3b8',
    marginHorizontal: 16,
    fontWeight: 'bold',
  },
  uploadButton: {
    backgroundColor: 'transparent',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#38bdf8',
    paddingVertical: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  uploadButtonText: {
    color: '#38bdf8',
    fontSize: 16,
    fontWeight: 'bold',
  },
  loadingMessage: {
    color: '#e2e8f0',
    textAlign: 'center',
    marginTop: 16,
    fontSize: 14,
  }
});
