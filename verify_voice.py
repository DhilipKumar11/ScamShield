import sys, os
sys.path.insert(0, 'backend')
sys.path.insert(0, 'backend/engines')
from engines.voice_analyzer import analyze_voice

tests = [
    ('Voice_Samples/processed/AI/AiGeneratedVoice.wav', 'AI'),
    ('Voice_Samples/processed/AI/Ai_gen_Voice.wav',     'AI'),
    ('Voice_Samples/processed/Human/HumanVoice1.wav',   'Human'),
    ('Voice_Samples/processed/Human/humanVoice.wav',    'Human'),
]

passed = 0
for path, expected in tests:
    r = analyze_voice(path)
    correct = (expected == 'AI' and r['verdict'] == 'AI_VOICE') or \
              (expected == 'Human' and r['verdict'] == 'HUMAN_VOICE')
    tag = 'PASS' if correct else 'FAIL'
    if correct:
        passed += 1
    print(f'[{tag}] {os.path.basename(path)}')
    print(f'       Expected : {expected}')
    print(f'       Verdict  : {r["verdict"]}  (conf={r["verdict_confidence"]:.1f}%)')
    print(f'       ML score : {r["ml_score"]}  KNN score: {r["knn_score"]}')
    print(f'       Ensemble : {r["synthetic_probability"]}')
    if r['nearest_matches']:
        print('       Top-3 nearest matches:')
        for m in r['nearest_matches'][:3]:
            print(f'         #{m["rank"]} {m["file"]:30s}  [{m["label"]:5s}]  {m["similarity"]:.1f}%')
    print()

print(f'Result: {passed}/{len(tests)} correct')
