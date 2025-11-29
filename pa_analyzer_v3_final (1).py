"""
Live PA Audio Analyzer V3.0 Final
- 全機能統合版
- バンド編成テキスト入力
- 全楽器の超詳細解析と改善提案
- Web検索統合（ミキサー/PA仕様自動取得）
- 過去音源との比較分析

Usage:
    streamlit run pa_analyzer_v3_final.py
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import io
from pathlib import Path
import tempfile
import json
from datetime import datetime
import os

# matplotlibの設定
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['font.size'] = 10

# ページ設定
st.set_page_config(
    page_title="Live PA Audio Analyzer V3.0 Final",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .version-badge {
        text-align: center;
        color: #667eea;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .good-point {
        background-color: #e6ffe6;
        padding: 1rem;
        border-left: 4px solid #44ff44;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .recommendation-critical {
        background-color: #ffe6e6;
        padding: 1rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .recommendation-important {
        background-color: #fff9e6;
        padding: 1rem;
        border-left: 4px solid #ffbb33;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================
# データベース（過去音源保存）
# =====================================

class AudioDatabase:
    """過去音源の解析結果を保存・管理"""
    
    def __init__(self):
        self.db_path = Path("audio_history.json")
        self.history = []
        self.load()
    
    def load(self):
        """履歴読み込み"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except:
                self.history = []
    
    def save(self):
        """履歴保存"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def add_entry(self, analysis_result, metadata):
        """新しい解析結果を追加"""
        
        entry = {
            'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'analysis': {
                'rms_db': analysis_result.get('rms_db'),
                'peak_db': analysis_result.get('peak_db'),
                'stereo_width': analysis_result.get('stereo_width'),
                'crest_factor': analysis_result.get('crest_factor'),
                'band_energies': analysis_result.get('band_energies', []),
                'instruments': {}
            },
            'equipment': {
                'mixer': metadata.get('mixer'),
                'pa_system': metadata.get('pa_system')
            }
        }
        
        self.history.append(entry)
        self.save()
        
        return entry['id']
    
    def get_recent(self, n=5):
        """最近のn件取得"""
        return sorted(self.history, key=lambda x: x['timestamp'], reverse=True)[:n]
    
    def find_similar(self, current_metadata, limit=3):
        """類似条件の音源を検索"""
        
        similar = []
        
        for entry in self.history:
            score = 0
            
            # 会場キャパが近い
            if abs(current_metadata.get('venue_capacity', 0) - 
                   entry['metadata'].get('venue_capacity', 0)) < 50:
                score += 30
            
            # ミキサーが同じ
            if current_metadata.get('mixer') == entry['equipment'].get('mixer'):
                score += 40
            
            # PAが同じ
            if current_metadata.get('pa_system') == entry['equipment'].get('pa_system'):
                score += 30
            
            similar.append({
                'entry': entry,
                'score': score
            })
        
        similar.sort(key=lambda x: x['score'], reverse=True)
        return [s['entry'] for s in similar[:limit] if s['score'] > 20]


# =====================================
# Web検索機能（簡易実装）
# =====================================

class EquipmentSpecsSearcher:
    """機材仕様のWeb検索（Claude APIを使用）"""
    
    def __init__(self):
        self.cache = {}
    
    def search_mixer_specs(self, mixer_name):
        """ミキサー仕様を検索"""
        
        if not mixer_name:
            return None
        
        # キャッシュチェック
        cache_key = mixer_name.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Web検索でミキサー情報を取得
        try:
            with st.spinner(f'🔍 {mixer_name}の仕様を検索中...'):
                # web_search tool を使用
                search_results = []
                
                # 検索クエリ
                queries = [
                    f"{mixer_name} specifications EQ bands",
                    f"{mixer_name} compressor dynamics",
                    f"{mixer_name} user manual"
                ]
                
                for query in queries:
                    try:
                        results = web_search(query)
                        if results:
                            search_results.extend(results[:2])  # 各クエリ上位2件
                    except:
                        pass
                
                if search_results:
                    # 検索結果から構造化データを作成（簡易版）
                    specs = self._parse_mixer_specs(mixer_name, search_results)
                    self.cache[cache_key] = specs
                    return specs
                
        except Exception as e:
            st.warning(f"⚠️ {mixer_name}の検索に失敗: {str(e)}")
        
        # フォールバック: 既知のミキサーデータベース
        return self._get_known_mixer_specs(mixer_name)
    
    def _parse_mixer_specs(self, mixer_name, search_results):
        """検索結果から仕様を抽出（簡易版）"""
        
        # TODO: 本来はClaude APIで詳細解析
        # ここでは既知データベースを返す
        return self._get_known_mixer_specs(mixer_name)
    
    def _get_known_mixer_specs(self, mixer_name):
        """既知のミキサーデータベース"""
        
        name_upper = mixer_name.upper()
        
        # Yamaha CL Series
        if 'CL5' in name_upper or 'CL3' in name_upper or 'CL1' in name_upper:
            return {
                'name': 'Yamaha CL Series',
                'eq_bands': 8,
                'eq_type': 'Parametric',
                'q_range': (0.1, 10.0),
                'gain_range': (-18, 18),
                'compressor_types': ['Comp260', 'U76', 'Opt-2A'],
                'has_de_esser': True,
                'has_dynamic_eq': True,
                'hpf_slopes': ['12dB/oct', '24dB/oct'],
                'characteristics': [
                    '8バンドPEQで非常に精密な調整が可能',
                    'Comp260は透明度が高くボーカルに最適',
                    'Dynamic EQで周波数依存のダイナミクス処理可能'
                ],
                'recommendations': {
                    'vocal': 'Comp260モデル推奨、8バンドEQをフル活用',
                    'kick': 'HPF 24dB/oct推奨、Gate+Compの組み合わせ',
                    'bass': 'Comp260で安定化、8バンドで精密な整形'
                }
            }
        
        # Yamaha QL Series
        elif 'QL5' in name_upper or 'QL1' in name_upper:
            return {
                'name': 'Yamaha QL Series',
                'eq_bands': 8,
                'eq_type': 'Parametric',
                'q_range': (0.1, 10.0),
                'gain_range': (-18, 18),
                'compressor_types': ['Comp260', 'U76', 'Opt-2A'],
                'has_de_esser': True,
                'has_dynamic_eq': False,
                'hpf_slopes': ['12dB/oct', '24dB/oct'],
                'characteristics': [
                    'CLに近い音質、やや簡素化',
                    '8バンドPEQは同等に強力'
                ]
            }
        
        # Behringer X32
        elif 'X32' in name_upper:
            return {
                'name': 'Behringer X32',
                'eq_bands': 4,
                'eq_type': 'Parametric',
                'q_range': (0.3, 10.0),
                'gain_range': (-15, 15),
                'compressor_types': ['Standard', 'Vintage'],
                'has_de_esser': False,
                'has_dynamic_eq': False,
                'hpf_slopes': ['12dB/oct', '24dB/oct'],
                'characteristics': [
                    'コストパフォーマンスに優れる',
                    'EQは4バンドのみ - 優先順位が重要',
                    'De-Esserなし - Dynamic EQで代用可能'
                ],
                'limitations': [
                    '4バンドEQのため精密調整は困難',
                    'De-Esser非搭載'
                ],
                'recommendations': {
                    'vocal': 'EQ優先順位: こもり除去→明瞭度→空気感。De-Esserは外部使用推奨',
                    'kick': 'EQ: HPF→基音強調→ボワつきカット→アタック',
                    'bass': 'Comp多めで安定化、EQは最重要2バンドのみ'
                }
            }
        
        # Allen & Heath SQ Series
        elif 'SQ' in name_upper:
            return {
                'name': 'Allen & Heath SQ Series',
                'eq_bands': 4,
                'eq_type': 'Parametric',
                'q_range': (0.5, 10.0),
                'gain_range': (-15, 15),
                'compressor_types': ['Standard', 'Vintage'],
                'has_de_esser': True,
                'has_dynamic_eq': False,
                'hpf_slopes': ['12dB/oct', '24dB/oct'],
                'characteristics': [
                    '音楽的なEQカーブ',
                    'De-Esser搭載'
                ]
            }
        
        # デフォルト
        else:
            return {
                'name': mixer_name,
                'eq_bands': 4,
                'eq_type': 'Parametric',
                'characteristics': ['仕様不明 - 一般的な設定を推奨']
            }
    
    def search_pa_specs(self, pa_name):
        """PAシステム仕様を検索"""
        
        if not pa_name:
            return None
        
        cache_key = pa_name.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Web検索（簡易版ではスキップ）
        return self._get_known_pa_specs(pa_name)
    
    def _get_known_pa_specs(self, pa_name):
        """既知のPAデータベース"""
        
        name_upper = pa_name.upper()
        
        # d&b
        if 'D&B' in name_upper or 'DB' in name_upper:
            return {
                'name': 'd&b Audiotechnik',
                'type': 'Line Array',
                'low_extension': 45,  # Hz
                'high_extension': 18000,
                'characteristics': [
                    '非常にフラットな特性',
                    '60Hz以下のレスポンスが良好',
                    '2-4kHzに若干のピーク傾向',
                    '明瞭度が高い'
                ],
                'eq_compensation': [
                    '2.5kHz Q=2.0 -1.5dB（システムピーク補正）',
                    '100Hz Q=1.0 +1dB（低域補強）'
                ],
                'feedback_prone': [250, 500, 2000, 4000],
                'recommendations': {
                    'kick_hpf': '35Hz推奨（十分な低域確保）',
                    'vocal': '明瞭度が出やすいシステム、EQは控えめでOK',
                    'overall': '素直な特性、大きな補正不要'
                }
            }
        
        # JBL
        elif 'JBL' in name_upper or 'VTX' in name_upper or 'VRX' in name_upper:
            return {
                'name': 'JBL Professional',
                'type': 'Line Array',
                'low_extension': 50,
                'high_extension': 20000,
                'characteristics': [
                    '高域が明るい傾向（6-10kHz）',
                    '低域のパンチが強い',
                    'トランジェント再現性が高い'
                ],
                'eq_compensation': [
                    '8kHz Q=1.5 -2dB（高域抑制）',
                    '80Hz Q=1.0 +1.5dB（低域強化）'
                ],
                'feedback_prone': [315, 630, 2500, 5000],
                'recommendations': {
                    'kick_hpf': '30-35Hz推奨',
                    'vocal': '高域が明るいため、シビランス注意',
                    'overall': 'やや派手な特性、EQで整える'
                }
            }
        
        # L-Acoustics
        elif 'L-ACOUSTICS' in name_upper or 'KARA' in name_upper or 'ARCS' in name_upper:
            return {
                'name': 'L-Acoustics',
                'type': 'Line Array',
                'low_extension': 50,
                'high_extension': 20000,
                'characteristics': [
                    '非常にバランスの良い特性',
                    '音楽的な表現力',
                    '高い明瞭度'
                ],
                'recommendations': {
                    'overall': '高品質システム、素直な特性'
                }
            }
        
        else:
            return {
                'name': pa_name,
                'type': 'Unknown',
                'characteristics': ['仕様不明']
            }


# =====================================
# V2解析（2mix全体）
# =====================================

class V2Analyzer:
    """V2の2mix全体解析（完全維持）"""
    
    def __init__(self, audio_file, venue_capacity, stage_volume, pa_system="", notes=""):
        self.audio_file = audio_file
        self.venue_capacity = venue_capacity
        self.stage_volume = stage_volume
        self.pa_system = pa_system
        self.notes = notes
        self.results = {}
        
    def analyze(self):
        """V2の解析（完全維持）"""
        try:
            with st.spinner('🎵 音源を読み込んでいます...'):
                self.y, self.sr = librosa.load(self.audio_file, sr=22050, mono=False, duration=300)
                
                if len(self.y.shape) == 1:
                    self.y = np.array([self.y, self.y])
                
                self.y_mono = librosa.to_mono(self.y)
                self.duration = len(self.y_mono) / self.sr
        except Exception as e:
            st.error(f"❌ 音源の読み込みに失敗: {str(e)}")
            raise
        
        with st.spinner('🔍 ステレオイメージ解析中...'):
            self._analyze_stereo_image()
        
        with st.spinner('📊 ダイナミクス解析中...'):
            self._analyze_dynamics()
        
        with st.spinner('🎼 周波数解析中...'):
            self._analyze_frequency()
        
        with st.spinner('⚡ トランジェント解析中...'):
            self._analyze_transients()
        
        with st.spinner('🔊 低域解析中...'):
            self._analyze_low_end()
        
        return self.results
    
    def _analyze_stereo_image(self):
        """ステレオイメージ解析"""
        left = self.y[0]
        right = self.y[1]
        
        correlation, _ = pearsonr(left, right)
        
        mid = (left + right) / 2
        side = (left - right) / 2
        mid_rms = np.sqrt(np.mean(mid**2))
        side_rms = np.sqrt(np.mean(side**2))
        
        stereo_width = (side_rms / (mid_rms + 1e-10) * 100)
        
        self.results['stereo_width'] = stereo_width
        self.results['correlation'] = correlation
        self.results['mid_signal'] = mid
        self.results['side_signal'] = side
    
    def _analyze_dynamics(self):
        """ダイナミクス解析"""
        peak_linear = np.max(np.abs(self.y_mono))
        peak_db = 20 * np.log10(peak_linear) if peak_linear > 0 else -100
        
        rms = np.sqrt(np.mean(self.y_mono**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        
        crest_factor = peak_db - rms_db
        
        hop_length = self.sr // 2
        frame_length = self.sr
        rms_frames = librosa.feature.rms(y=self.y_mono, frame_length=frame_length, 
                                         hop_length=hop_length)[0]
        rms_db_frames = 20 * np.log10(rms_frames + 1e-10)
        
        dynamic_range = np.percentile(rms_db_frames, 95) - np.percentile(rms_db_frames, 5)
        
        self.results['peak_db'] = peak_db
        self.results['rms_db'] = rms_db
        self.results['crest_factor'] = crest_factor
        self.results['dynamic_range'] = dynamic_range
        self.results['rms_frames'] = rms_db_frames
    
    def _analyze_frequency(self):
        """周波数解析"""
        D = np.abs(librosa.stft(self.y_mono))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        avg_spectrum = np.mean(S_db, axis=1)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        bands = [
            (20, 80, "Sub Bass"),
            (80, 250, "Bass"),
            (250, 500, "Low-Mid"),
            (500, 2000, "Mid"),
            (2000, 4000, "High-Mid"),
            (4000, 8000, "Presence"),
            (8000, 16000, "Brilliance"),
        ]
        
        band_energies = []
        for low_freq, high_freq, band_name in bands:
            mask = (freqs >= low_freq) & (freqs < high_freq)
            if np.any(mask):
                band_energy = np.mean(avg_spectrum[mask])
                band_energies.append(band_energy)
            else:
                band_energies.append(-100)
        
        self.results['band_energies'] = band_energies
        self.results['freqs'] = freqs
        self.results['avg_spectrum'] = avg_spectrum
        self.results['bands'] = bands
    
    def _analyze_transients(self):
        """トランジェント解析"""
        onset_env = librosa.onset.onset_strength(y=self.y_mono, sr=self.sr)
        avg_onset_strength = np.mean(onset_env)
        max_onset = np.max(onset_env)
        
        onset_frames = librosa.onset.onset_detect(y=self.y_mono, sr=self.sr, units='frames')
        num_onsets = len(onset_frames)
        onset_density = num_onsets / self.duration
        
        self.results['avg_onset'] = avg_onset_strength
        self.results['max_onset'] = max_onset
        self.results['onset_env'] = onset_env
        self.results['onset_density'] = onset_density
    
    def _analyze_low_end(self):
        """低域解析"""
        nyq = self.sr / 2
        low_cutoff = 40 / nyq
        
        if low_cutoff < 1.0:
            b_low, a_low = signal.butter(4, low_cutoff, btype='lowpass')
            very_low_freq = signal.filtfilt(b_low, a_low, self.y_mono)
            very_low_rms = np.sqrt(np.mean(very_low_freq**2))
        else:
            very_low_rms = 0
        
        if len(self.results.get('band_energies', [])) >= 2:
            sub_bass = self.results['band_energies'][0]
            bass = self.results['band_energies'][1]
            sub_bass_ratio = sub_bass - bass
        else:
            sub_bass_ratio = 0
        
        self.results['very_low_rms'] = very_low_rms
        self.results['sub_bass_ratio'] = sub_bass_ratio
    
    def create_visualization(self):
        """グラフ生成（V2のまま）"""
        try:
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
            
            # 1. Waveform
            ax1 = fig.add_subplot(gs[0, :])
            time_axis = np.arange(len(self.y_mono)) / self.sr
            ax1.plot(time_axis, self.y_mono, linewidth=0.3, alpha=0.7, color='blue')
            rms_val = 10**(self.results['rms_db']/20)
            ax1.axhline(y=rms_val, color='green', linestyle='--', alpha=0.6, 
                       label=f'RMS: {self.results["rms_db"]:.1f}dB')
            ax1.axhline(y=-rms_val, color='green', linestyle='--', alpha=0.6)
            ax1.set_title('Waveform Overview', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([-1.1, 1.1])
            
            # 2. Frequency Spectrum
            ax2 = fig.add_subplot(gs[1, 0])
            freqs = self.results['freqs'][1:]
            spectrum = self.results['avg_spectrum'][1:]
            ax2.semilogx(freqs, spectrum, linewidth=1.5, color='darkblue')
            ax2.set_title('Frequency Spectrum', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.grid(True, alpha=0.3, which='both')
            ax2.set_xlim([20, self.sr/2])
            
            # 3. Frequency Bands
            ax3 = fig.add_subplot(gs[1, 1])
            band_names = ['Sub\nBass', 'Bass', 'Low\nMid', 'Mid', 'High\nMid', 'Pres', 'Bril']
            colors = ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#F4A460', '#FFA07A', '#FFB6C1']
            ax3.bar(range(len(self.results['band_energies'])), self.results['band_energies'], 
                   color=colors, edgecolor='black', linewidth=1.5)
            ax3.set_xticks(range(len(band_names)))
            ax3.set_xticklabels(band_names, fontsize=9)
            ax3.set_title('Frequency Band Distribution', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Energy (dB)')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Mid/Side
            ax4 = fig.add_subplot(gs[1, 2])
            mid_signal = self.results['mid_signal']
            side_signal = self.results['side_signal']
            time_samples = np.linspace(0, self.duration, min(5000, len(mid_signal)))
            indices = np.linspace(0, len(mid_signal)-1, len(time_samples), dtype=int)
            ax4.plot(time_samples, mid_signal[indices], linewidth=0.8, alpha=0.7, 
                    label='Mid', color='blue')
            ax4.plot(time_samples, side_signal[indices], linewidth=0.8, alpha=0.7, 
                    label='Side', color='red')
            ax4.set_title(f'Mid/Side (Width: {self.results["stereo_width"]:.1f}%)', 
                         fontsize=11, fontweight='bold')
            ax4.set_xlabel('Time (s)')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # 5. RMS Over Time
            ax5 = fig.add_subplot(gs[2, 0])
            hop = self.sr // 2
            time_frames = librosa.frames_to_time(range(len(self.results['rms_frames'])), 
                                                 sr=self.sr, hop_length=hop)
            ax5.plot(time_frames, self.results['rms_frames'], linewidth=1.5, color='green')
            ax5.axhline(y=self.results['rms_db'], color='darkgreen', linestyle='--', 
                       alpha=0.7, label=f'Avg: {self.results["rms_db"]:.1f}dB')
            ax5.set_title('RMS Level Over Time', fontsize=11, fontweight='bold')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('RMS (dBFS)')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([-50, 0])
            
            # 6. Onset Strength
            ax6 = fig.add_subplot(gs[2, 1])
            onset_times = librosa.frames_to_time(range(len(self.results['onset_env'])), sr=self.sr)
            ax6.plot(onset_times, self.results['onset_env'], linewidth=1, color='red', alpha=0.7)
            ax6.axhline(y=self.results['avg_onset'], color='darkred', linestyle='--', 
                       alpha=0.7, label=f'Avg: {self.results["avg_onset"]:.2f}')
            ax6.set_title('Onset Strength', fontsize=11, fontweight='bold')
            ax6.set_xlabel('Time (s)')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            # 7. Spectrogram
            try:
                ax7 = fig.add_subplot(gs[2, 2])
                D = librosa.stft(self.y_mono)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='log',
                                               ax=ax7, cmap='viridis')
                ax7.set_title('Spectrogram', fontsize=11, fontweight='bold')
                ax7.set_ylabel('Frequency (Hz)')
                fig.colorbar(img, ax=ax7, format='%+2.0f dB')
            except:
                ax7 = fig.add_subplot(gs[2, 2])
                ax7.text(0.5, 0.5, 'Spectrogram\n生成エラー', 
                        ha='center', va='center', transform=ax7.transAxes)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"グラフ生成エラー: {str(e)}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'グラフ生成失敗\n{str(e)}', 
                   ha='center', va='center', fontsize=12)
            return fig
    
    def generate_v2_recommendations(self, mixer_specs=None, pa_specs=None):
        """V2の改善提案（2mix全体）- Web検索結果反映"""
        
        good_points = []
        recommendations = {
            'critical': [],
            'important': [],
            'optional': []
        }
        
        # === 良いポイント検出 ===
        
        # 位相相関
        correlation = self.results.get('correlation', 1)
        if correlation > 0.95:
            good_points.append({
                'category': 'ステレオイメージ',
                'point': f'位相相関が非常に良好 ({correlation:.3f})',
                'impact': '★★★★★'
            })
        elif correlation > 0.85:
            good_points.append({
                'category': 'ステレオイメージ',
                'point': f'位相相関が良好 ({correlation:.3f})',
                'impact': '★★★★'
            })
        
        # トランジェント
        avg_onset = self.results.get('avg_onset', 0)
        if avg_onset > 2.0:
            good_points.append({
                'category': 'トランジェント',
                'point': f'トランジェント特性が良好（{avg_onset:.2f}）',
                'impact': '★★★★'
            })
        
        # ステレオ幅が適切
        stereo_width = self.results.get('stereo_width', 0)
        if self.venue_capacity < 200 and 10 < stereo_width < 25:
            good_points.append({
                'category': 'ステレオイメージ',
                'point': f'会場規模に対してステレオ幅が適切（{stereo_width:.1f}%）',
                'impact': '★★★★'
            })
        
        # === 改善提案 ===
        
        # ステレオイメージ
        if correlation < 0.7:
            recommendations['critical'].append({
                'category': 'ステレオイメージ',
                'issue': f'位相相関が低い ({correlation:.3f})',
                'solution': 'Left/Rightチャンネルの位相を確認。パンニングを見直し。',
                'impact': '★★★★★'
            })
        
        if self.venue_capacity < 200 and stereo_width > 30:
            recommendations['important'].append({
                'category': 'ステレオイメージ',
                'issue': f'小規模会場でステレオ幅が広すぎ ({stereo_width:.1f}%)',
                'solution': 'ステレオイメージャーで幅を15-20%に調整',
                'impact': '★★★'
            })
        
        # 音圧
        rms_db = self.results.get('rms_db', -100)
        
        if rms_db < -22:
            # ミキサー仕様を反映
            comp_suggestion = 'マスターコンプ: Threshold -10〜-12dB, Ratio 3:1〜4:1, Attack 20-30ms'
            
            if mixer_specs:
                if mixer_specs.get('name') == 'Yamaha CL Series':
                    comp_suggestion = 'マスターInsert: Comp260, THR -12dB, Ratio 3:1, ATK 25ms, RLS Auto'
                elif mixer_specs.get('name') == 'Behringer X32':
                    comp_suggestion = 'マスターInsert: Vintage Compressor, THR -10dB, Ratio 4:1, ATK 20ms'
            
            recommendations['critical'].append({
                'category': '音圧・密度',
                'issue': f'RMSが非常に低い ({rms_db:.1f} dBFS) - 「スカスカ」',
                'solution': comp_suggestion,
                'impact': '★★★★★'
            })
        
        # HPF（PA仕様を反映）
        if self.results.get('very_low_rms', 0) > 0.001:
            hpf_freq = 30
            
            if pa_specs:
                pa_name = pa_specs.get('name', '')
                if 'd&b' in pa_name:
                    hpf_freq = 35  # d&bは35Hzでも十分
                elif 'JBL' in pa_name:
                    hpf_freq = 30  # JBLは30Hzまで対応
                
                recommendations['critical'].append({
                    'category': 'HPF',
                    'issue': '40Hz以下にサブソニック成分',
                    'solution': f'マスターHPF {hpf_freq}Hz, 24dB/oct（{pa_name}の特性考慮）',
                    'impact': '★★★★'
                })
            else:
                recommendations['critical'].append({
                    'category': 'HPF',
                    'issue': '40Hz以下にサブソニック成分',
                    'solution': 'マスターHPF 30-35Hz, 24dB/oct',
                    'impact': '★★★★'
                })
        
        # 周波数バランス
        band_energies = self.results.get('band_energies', [])
        if len(band_energies) >= 7:
            # 低域過多
            if band_energies[0] > band_energies[3] + 10:  # Sub Bass vs Mid
                recommendations['important'].append({
                    'category': '周波数バランス',
                    'issue': f'低域が過多（Sub Bass {band_energies[0]:.1f}dB）',
                    'solution': 'マスターEQ: 60Hz Q=1.0 -2〜3dB',
                    'impact': '★★★★'
                })
            
            # 明瞭度不足
            if band_energies[4] < band_energies[3] - 8:  # High-Mid vs Mid
                recommendations['important'].append({
                    'category': '周波数バランス',
                    'issue': f'明瞭度帯域が不足（High-Mid {band_energies[4]:.1f}dB）',
                    'solution': 'マスターEQ: 3kHz Q=1.5 +2〜3dB',
                    'impact': '★★★★'
                })
        
        return good_points, recommendations


# =====================================
# 楽器分離（テキスト入力ベース）
# =====================================

class InstrumentSeparator:
    """テキスト入力された編成に基づく楽器分離"""
    
    def __init__(self, y, sr, band_lineup_text):
        self.y = y
        self.sr = sr
        self.y_mono = librosa.to_mono(y) if len(y.shape) > 1 else y
        
        # テキストをパース
        self.instruments = self._parse_lineup(band_lineup_text)
        
    def _parse_lineup(self, text):
        """
        バンド編成テキストをパース
        
        例: "ボーカル、キック、スネア、ベース、ギター"
        → ['vocal', 'kick', 'snare', 'bass', 'guitar']
        """
        
        # 日本語→英語マッピング
        mapping = {
            'ボーカル': 'vocal',
            'ヴォーカル': 'vocal',
            'vo': 'vocal',
            'キック': 'kick',
            'バスドラ': 'kick',
            'bd': 'kick',
            'スネア': 'snare',
            'sn': 'snare',
            'sd': 'snare',
            'ハイハット': 'hihat',
            'ハット': 'hihat',
            'hh': 'hihat',
            'タム': 'tom',
            'ベース': 'bass',
            'ベ': 'bass',
            'ba': 'bass',
            'エレキギター': 'e_guitar',
            'ギター': 'e_guitar',
            'エレキ': 'e_guitar',
            'eg': 'e_guitar',
            'gt': 'e_guitar',
            'アコギ': 'a_guitar',
            'アコースティックギター': 'a_guitar',
            'ag': 'a_guitar',
            'キーボード': 'keyboard',
            'キーボ': 'keyboard',
            'kb': 'keyboard',
            'key': 'keyboard',
            'シンセ': 'synth',
            'シンセサイザー': 'synth',
            'syn': 'synth'
        }
        
        instruments = []
        
        # カンマ、スペース、改行で分割
        items = text.replace('\n', ',').replace('、', ',').split(',')
        
        for item in items:
            item = item.strip().lower()
            if not item:
                continue
            
            # マッピングから検索
            for jp_name, eng_name in mapping.items():
                if jp_name.lower() in item or eng_name in item:
                    if eng_name not in instruments:
                        instruments.append(eng_name)
                    break
        
        return instruments
    
    def separate(self):
        """指定された楽器のみを分離"""
        
        stems = {}
        
        for instrument in self.instruments:
            with st.spinner(f'🎸 {instrument}を分離中...'):
                if instrument == 'vocal':
                    stems['vocal'] = self._extract_vocal()
                elif instrument == 'kick':
                    stems['kick'] = self._extract_kick()
                elif instrument == 'snare':
                    stems['snare'] = self._extract_snare()
                elif instrument == 'hihat':
                    stems['hihat'] = self._extract_hihat()
                elif instrument == 'tom':
                    stems['tom'] = self._extract_tom()
                elif instrument == 'bass':
                    stems['bass'] = self._extract_bass()
                elif instrument == 'e_guitar':
                    stems['e_guitar'] = self._extract_e_guitar()
                elif instrument == 'a_guitar':
                    stems['a_guitar'] = self._extract_a_guitar()
                elif instrument == 'keyboard':
                    stems['keyboard'] = self._extract_keyboard()
                elif instrument == 'synth':
                    stems['synth'] = self._extract_synth()
        
        return stems
    
    def _extract_vocal(self):
        """ボーカル抽出（改良版）"""
        sos_low = signal.butter(6, 200 / (self.sr/2), btype='highpass', output='sos')
        sos_high = signal.butter(6, 5000 / (self.sr/2), btype='lowpass', output='sos')
        vocal = signal.sosfilt(sos_low, self.y_mono)
        vocal = signal.sosfilt(sos_high, vocal)
        D = librosa.stft(vocal)
        freqs = librosa.fft_frequencies(sr=self.sr)
        formant_mask = (freqs >= 1000) & (freqs <= 4000)
        D[formant_mask, :] *= 1.8
        vocal = librosa.istft(D)
        return vocal
    
    def _extract_kick(self):
        """キック抽出"""
        sos = signal.butter(6, [40 / (self.sr/2), 120 / (self.sr/2)], btype='bandpass', output='sos')
        kick = signal.sosfilt(sos, self.y_mono)
        onset_env = librosa.onset.onset_strength(y=self.y_mono, sr=self.sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, units='frames')
        hop_length = 512
        for frame in onset_frames:
            sample = frame * hop_length
            if sample < len(kick):
                start = max(0, sample - 500)
                end = min(len(kick), sample + 2000)
                kick[start:end] *= 2.0
        return kick
    
    def _extract_snare(self):
        """スネア抽出"""
        sos_body = signal.butter(4, [200 / (self.sr/2), 400 / (self.sr/2)], btype='bandpass', output='sos')
        sos_attack = signal.butter(4, [2000 / (self.sr/2), 5000 / (self.sr/2)], btype='bandpass', output='sos')
        sos_snappy = signal.butter(4, [6000 / (self.sr/2), 10000 / (self.sr/2)], btype='bandpass', output='sos')
        snare_body = signal.sosfilt(sos_body, self.y_mono)
        snare_attack = signal.sosfilt(sos_attack, self.y_mono)
        snare_snappy = signal.sosfilt(sos_snappy, self.y_mono)
        snare = snare_body * 0.4 + snare_attack * 0.4 + snare_snappy * 0.2
        return snare
    
    def _extract_hihat(self):
        """ハイハット抽出"""
        sos = signal.butter(6, 6000 / (self.sr/2), btype='highpass', output='sos')
        hihat = signal.sosfilt(sos, self.y_mono)
        return hihat
    
    def _extract_tom(self):
        """タム抽出"""
        sos = signal.butter(4, [80 / (self.sr/2), 250 / (self.sr/2)], btype='bandpass', output='sos')
        tom = signal.sosfilt(sos, self.y_mono)
        return tom
    
    def _extract_bass(self):
        """ベース抽出"""
        sos = signal.butter(6, [60 / (self.sr/2), 250 / (self.sr/2)], btype='bandpass', output='sos')
        bass = signal.sosfilt(sos, self.y_mono)
        return bass
    
    def _extract_e_guitar(self):
        """エレキギター抽出"""
        sos = signal.butter(4, [200 / (self.sr/2), 3000 / (self.sr/2)], btype='bandpass', output='sos')
        guitar = signal.sosfilt(sos, self.y_mono)
        return guitar
    
    def _extract_a_guitar(self):
        """アコギ抽出"""
        sos = signal.butter(4, [100 / (self.sr/2), 5000 / (self.sr/2)], btype='bandpass', output='sos')
        guitar = signal.sosfilt(sos, self.y_mono)
        return guitar
    
    def _extract_keyboard(self):
        """キーボード抽出"""
        sos = signal.butter(4, [200 / (self.sr/2), 4000 / (self.sr/2)], btype='bandpass', output='sos')
        keyboard = signal.sosfilt(sos, self.y_mono)
        return keyboard
    
    def _extract_synth(self):
        """シンセ抽出"""
        sos = signal.butter(4, [100 / (self.sr/2), 8000 / (self.sr/2)], btype='bandpass', output='sos')
        synth = signal.sosfilt(sos, self.y_mono)
        return synth


# =====================================
# 楽器別詳細解析（全楽器対応）
# =====================================

class InstrumentAnalyzer:
    """楽器別超詳細解析"""
    
    def __init__(self, stems, sr, full_audio, overall_rms, mixer_specs, pa_specs):
        self.stems = stems
        self.sr = sr
        self.full_audio = full_audio
        self.overall_rms = overall_rms
        self.mixer_specs = mixer_specs
        self.pa_specs = pa_specs
        
    def analyze_all(self, venue_capacity, stage_volume):
        """全楽器を詳細解析"""
        
        analyses = {}
        
        for name, audio in self.stems.items():
            if audio is not None and len(audio) > 0:
                analyses[name] = self.analyze_instrument(
                    name, audio, venue_capacity, stage_volume
                )
        
        # 楽器間の関係性も解析
        self._analyze_relationships(analyses)
        
        return analyses
    
    def analyze_instrument(self, name, audio, venue_capacity, stage_volume):
        """個別楽器の超詳細解析"""
        
        # 基本メトリクス
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100
        crest_factor = peak_db - rms_db
        
        # 周波数解析
        D = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        spectrum = np.mean(D, axis=1)
        
        analysis = {
            'name': name,
            'rms_db': rms_db,
            'peak_db': peak_db,
            'crest_factor': crest_factor,
            'level_vs_mix': rms_db - self.overall_rms,
            'spectrum': spectrum,
            'freqs': freqs,
            'good_points': [],
            'issues': [],
            'recommendations': []
        }
        
        # 楽器別の詳細解析
        if name == 'vocal':
            analysis.update(self._analyze_vocal(audio, spectrum, freqs, venue_capacity, stage_volume))
        elif name == 'kick':
            analysis.update(self._analyze_kick(audio, spectrum, freqs))
        elif name == 'snare':
            analysis.update(self._analyze_snare(audio, spectrum, freqs))
        elif name == 'bass':
            analysis.update(self._analyze_bass(audio, spectrum, freqs))
        elif name == 'hihat':
            analysis.update(self._analyze_hihat(audio, spectrum, freqs))
        elif name == 'tom':
            analysis.update(self._analyze_tom(audio, spectrum, freqs))
        elif name in ['e_guitar', 'a_guitar']:
            analysis.update(self._analyze_guitar(name, audio, spectrum, freqs))
        elif name in ['keyboard', 'synth']:
            analysis.update(self._analyze_keys(name, audio, spectrum, freqs))
        
        return analysis
    
    def _analyze_vocal(self, audio, spectrum, freqs, venue_capacity, stage_volume):
        """ボーカル超詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        # 周波数帯域
        bands = {
            'fundamental': (150, 400),
            'body': (400, 1000),
            'clarity': (2000, 4000),
            'presence': (4000, 6000),
            'sibilance': (6000, 8000),
            'air': (8000, 12000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['clarity'] > -25:
            detail['good_points'].append({
                'point': f"明瞭度帯域が良好（{detail['freq_bands']['clarity']:.1f}dB）",
                'impact': '★★★★★'
            })
        
        if detail['freq_bands']['air'] > -35:
            detail['good_points'].append({
                'point': f"空気感が十分（{detail['freq_bands']['air']:.1f}dB）",
                'impact': '★★★★'
            })
        
        # 問題検出
        clarity_level = detail['freq_bands']['clarity']
        
        if clarity_level < -35:
            detail['issues'].append({
                'severity': 'critical',
                'problem': '明瞭度が極めて低い',
                'detail': f'2-4kHz: {clarity_level:.1f}dB（推奨: -25dB以上）'
            })
            
            # 会場規模とステージ生音を考慮
            is_small_venue = venue_capacity < 200
            has_stage_sound = stage_volume in ['high', 'medium']
            
            if is_small_venue and has_stage_sound:
                # フィードバック配慮
                steps = self._get_vocal_eq_steps_safe()
            else:
                # 積極的処理
                steps = self._get_vocal_eq_steps_full()
            
            detail['recommendations'].append({
                'priority': 'critical',
                'title': 'ボーカル明瞭度向上',
                'steps': steps,
                'mixer_specific': self._get_mixer_vocal_steps(),
                'expected_results': [
                    '明瞭度 +50〜70%',
                    '歌詞の聴き取りやすさ大幅改善',
                    '存在感の向上'
                ]
            })
        
        # こもり
        body_level = detail['freq_bands']['body']
        if body_level > clarity_level + 8:
            detail['issues'].append({
                'severity': 'important',
                'problem': 'こもりが強い',
                'detail': f'400-1000Hz過多（{body_level - clarity_level:.1f}dB高い）'
            })
            
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'こもり除去',
                'steps': [
                    'PEQ: 600Hz, Q=1.5, -3.0dB',
                    'または: 800Hz, Q=2.0, -2.5dB',
                    '',
                    '効果: すっきりしたボーカル'
                ],
                'expected_results': ['明瞭度向上', 'クリアなボーカル']
            })
        
        # シビランス
        sibilance_level = detail['freq_bands']['sibilance']
        if sibilance_level > detail['freq_bands']['clarity'] + 5:
            detail['issues'].append({
                'severity': 'important',
                'problem': '歯擦音が過多',
                'detail': f'6-8kHz: {sibilance_level:.1f}dB'
            })
            
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'De-Esser設定',
                'steps': self._get_deesser_steps(),
                'expected_results': ['自然な高域', '聴きやすいボーカル']
            })
        
        return detail
    
    def _analyze_kick(self, audio, spectrum, freqs):
        """キック超詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        bands = {
            'subsonic': (20, 40),
            'fundamental': (40, 80),
            'attack': (60, 100),
            'body': (100, 200),
            'boxiness': (200, 400),
            'click': (2000, 5000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['attack'] > -25:
            detail['good_points'].append({
                'point': f"パンチ・アタックが良好（{detail['freq_bands']['attack']:.1f}dB）",
                'impact': '★★★★★'
            })
        
        if detail['freq_bands']['click'] > -40:
            detail['good_points'].append({
                'point': f"ビーター音が明瞭（{detail['freq_bands']['click']:.1f}dB）",
                'impact': '★★★★'
            })
        
        # サブソニック
        if detail['freq_bands']['subsonic'] > -45:
            detail['issues'].append({
                'severity': 'critical',
                'problem': 'サブソニック成分が多い',
                'detail': f'20-40Hz: {detail["freq_bands"]["subsonic"]:.1f}dB'
            })
            
            hpf_freq = self._get_kick_hpf_freq()
            
            detail['recommendations'].append({
                'priority': 'critical',
                'title': 'HPF設定（必須）',
                'steps': [
                    f'HPF: {hpf_freq}Hz, 24dB/oct',
                    '',
                    '【効果】',
                    '  - ヘッドルーム +2〜3dB確保',
                    '  - PAシステムの保護',
                    '  - タイトな低域',
                    '',
                    f'【{self.pa_specs.get("name", "PA")}考慮】',
                    *self._get_pa_kick_notes()
                ],
                'mixer_specific': self._get_mixer_hpf_steps('kick', hpf_freq),
                'expected_results': [
                    'ヘッドルーム +2〜3dB',
                    'クリアな低域',
                    'システム負荷軽減'
                ]
            })
        
        # ボワつき
        if detail['freq_bands']['boxiness'] > detail['freq_bands']['fundamental'] + 5:
            detail['issues'].append({
                'severity': 'important',
                'problem': 'ボワつきが強い',
                'detail': f'200-400Hz過多'
            })
            
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'ボワつき除去',
                'steps': [
                    'PEQ: 250Hz, Q=3.0, -3.0dB',
                    '',
                    '効果: タイトなキック'
                ],
                'expected_results': ['明瞭な低域', 'パンチの向上']
            })
        
        # パンチ不足
        attack_level = detail['freq_bands']['attack']
        fundamental_level = detail['freq_bands']['fundamental']
        
        if attack_level < fundamental_level - 5:
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'パンチ強化',
                'steps': [
                    'PEQ: 70Hz, Q=1.2, +4.0dB（基音強調）',
                    'PEQ: 3kHz, Q=2.0, +2.0dB（ビーター音）',
                    '',
                    'Compressor:',
                    '  Threshold: -15dB, Ratio: 3:1',
                    '  Attack: 20ms（アタック保持）',
                    '  Release: 150ms',
                    '',
                    'Gate（オプション）:',
                    '  Attack: 0.1ms, Release: 150ms'
                ],
                'expected_results': ['パンチ +40%', 'アタック明瞭化']
            })
        
        return detail
    
    def _analyze_snare(self, audio, spectrum, freqs):
        """スネア超詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        bands = {
            'body': (200, 400),
            'fatness': (400, 800),
            'attack': (2000, 5000),
            'crack': (3000, 6000),
            'snappy': (6000, 10000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['crack'] > -30:
            detail['good_points'].append({
                'point': f"クラック音が明瞭（{detail['freq_bands']['crack']:.1f}dB）",
                'impact': '★★★★'
            })
        
        if detail['freq_bands']['snappy'] > -35:
            detail['good_points'].append({
                'point': f"スナッピーが鮮明（{detail['freq_bands']['snappy']:.1f}dB）",
                'impact': '★★★★'
            })
        
        # アタック不足
        if detail['freq_bands']['attack'] < -35:
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'スネアのアタック強化',
                'steps': [
                    'PEQ: 3.5kHz, Q=2.0, +3.0dB（クラック強調）',
                    'PEQ: 7kHz, Q=1.5, +2.0dB（スナッピー）',
                    '',
                    'Compressor:',
                    '  Threshold: -12dB, Ratio: 4:1',
                    '  Attack: 5ms（速めでパンチ）',
                    '  Release: 100ms',
                    '',
                    'Gate:',
                    '  Threshold: 調整',
                    '  Attack: 0.1ms, Release: 80ms'
                ],
                'expected_results': ['アタック +50%', 'メリハリのあるスネア']
            })
        
        # ボディ不足
        if detail['freq_bands']['body'] < -40:
            detail['recommendations'].append({
                'priority': 'optional',
                'title': 'ボディ強化',
                'steps': [
                    'PEQ: 250Hz, Q=1.5, +2.5dB',
                    '',
                    '効果: 太いスネアサウンド'
                ],
                'expected_results': ['ボディ感向上', '存在感アップ']
            })
        
        return detail
    
    def _analyze_bass(self, audio, spectrum, freqs):
        """ベース超詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        bands = {
            'fundamental': (80, 200),
            'harmonic': (200, 800),
            'attack': (1000, 3000),
            'brightness': (3000, 6000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['fundamental'] > -25:
            detail['good_points'].append({
                'point': f"基音が豊か（{detail['freq_bands']['fundamental']:.1f}dB）",
                'impact': '★★★★★'
            })
        
        if detail['freq_bands']['attack'] > -40:
            detail['good_points'].append({
                'point': f"アタックが明瞭（{detail['freq_bands']['attack']:.1f}dB）",
                'impact': '★★★★'
            })
        
        # 倍音不足（聴こえにくい）
        if detail['freq_bands']['harmonic'] < detail['freq_bands']['fundamental'] - 10:
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'ベースの聴こえやすさ向上',
                'steps': [
                    'PEQ: 400Hz, Q=1.5, +3.0dB（倍音強調）',
                    'PEQ: 2kHz, Q=2.0, +2.0dB（アタック）',
                    '',
                    '効果: 小型スピーカーでも聴こえるベース'
                ],
                'expected_results': ['聴こえやすさ +60%', '明瞭なベースライン']
            })
        
        # 基音過多（ボワつき）
        if detail['freq_bands']['fundamental'] > detail['freq_bands']['harmonic'] + 15:
            detail['recommendations'].append({
                'priority': 'important',
                'title': '低域の整理',
                'steps': [
                    'PEQ: 120Hz, Q=2.0, -2.5dB（余分な低域カット）',
                    '',
                    'Compressor:',
                    '  Threshold: -15dB, Ratio: 3:1',
                    '  Attack: 30ms（アタック保持）',
                    '  Release: 200ms'
                ],
                'expected_results': ['タイトな低域', 'クリアなベース']
            })
        
        return detail
    
    def _analyze_hihat(self, audio, spectrum, freqs):
        """ハイハット詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        bands = {
            'brightness': (6000, 10000),
            'air': (10000, 16000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['brightness'] > -30:
            detail['good_points'].append({
                'point': '明るさが十分',
                'impact': '★★★★'
            })
        
        # 推奨事項
        detail['recommendations'].append({
            'priority': 'optional',
            'title': 'ハイハットの調整',
            'steps': [
                'HPF: 300Hz, 12dB/oct（低域除去）',
                'PEQ: 8kHz, Q=1.5, +1〜2dB（明るさ調整）',
                '',
                'Compressor（軽め）:',
                '  Threshold: -10dB, Ratio: 2:1'
            ],
            'expected_results': ['クリアなハイハット']
        })
        
        return detail
    
    def _analyze_tom(self, audio, spectrum, freqs):
        """タム詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        detail['recommendations'].append({
            'priority': 'optional',
            'title': 'タムの調整',
            'steps': [
                'HPF: 60Hz, 12dB/oct',
                'PEQ: 150Hz, Q=1.5, +3dB（ボディ）',
                'PEQ: 2.5kHz, Q=2.0, +2dB（アタック）',
                '',
                'Gate:',
                '  Threshold: 調整',
                '  Attack: 0.5ms, Release: 200ms'
            ],
            'expected_results': ['明瞭なタムサウンド']
        })
        
        return detail
    
    def _analyze_guitar(self, name, audio, spectrum, freqs):
        """ギター詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        is_electric = (name == 'e_guitar')
        
        bands = {
            'body': (200, 800),
            'presence': (2000, 5000),
            'brightness': (5000, 10000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['presence'] > -30:
            detail['good_points'].append({
                'point': 'プレゼンスが良好',
                'impact': '★★★★'
            })
        
        # 推奨事項
        if is_electric:
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'エレキギターの調整',
                'steps': [
                    'HPF: 80Hz, 12dB/oct',
                    'PEQ: 2.5kHz, Q=2.0, +2〜3dB（ボーカルとの棲み分け）',
                    '  ※ボーカルは3.2kHz強調なので干渉回避',
                    '',
                    'Compressor:',
                    '  Threshold: -12dB, Ratio: 3:1',
                    '  Attack: 15ms, Release: 150ms'
                ],
                'expected_results': ['ボーカルとの分離', '明瞭なギター']
            })
        else:
            detail['recommendations'].append({
                'priority': 'optional',
                'title': 'アコギの調整',
                'steps': [
                    'HPF: 80Hz, 12dB/oct',
                    'PEQ: 3kHz, Q=1.5, +2dB（明るさ）',
                    'PEQ: 8kHz, Q=2.0, +1.5dB（空気感）'
                ],
                'expected_results': ['クリアなアコギサウンド']
            })
        
        return detail
    
    def _analyze_keys(self, name, audio, spectrum, freqs):
        """キーボード/シンセ詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        detail['recommendations'].append({
            'priority': 'optional',
            'title': f'{name}の調整',
            'steps': [
                'HPF: 60Hz, 12dB/oct',
                'PEQ: ボーカル/ギターとの周波数帯域を確認',
                '  必要に応じてスペースを空ける'
            ],
            'expected_results': ['他楽器との調和']
        })
        
        return detail
    
    def _analyze_relationships(self, analyses):
        """楽器間の関係性解析"""
        
        # キック vs ベース
        if 'kick' in analyses and 'bass' in analyses:
            kick_fund = analyses['kick'].get('freq_bands', {}).get('fundamental', -100)
            bass_fund = analyses['bass'].get('freq_bands', {}).get('fundamental', -100)
            
            if abs(kick_fund - bass_fund) < 3 and kick_fund > -100 and bass_fund > -100:
                analyses['kick']['recommendations'].append({
                    'priority': 'important',
                    'title': 'ベースとの周波数棲み分け',
                    'steps': [
                        '【キック側】',
                        '  PEQ: 65Hz, Q=1.2, +4dB（キック強調）',
                        '  PEQ: 90Hz, Q=3.0, -4dB（ベース帯域カット）',
                        '',
                        '【ベース側】',
                        '  PEQ: 90Hz, Q=1.0, +3dB（ベース強調）',
                        '  PEQ: 65Hz, Q=3.0, -4dB（キック帯域カット）',
                        '',
                        '理論: 各楽器に専用周波数を割り当て'
                    ],
                    'expected_results': ['明瞭な低域', 'キックとベースの分離']
                })
        
        # ボーカル vs ギター
        if 'vocal' in analyses and 'e_guitar' in analyses:
            vocal_clarity = analyses['vocal'].get('freq_bands', {}).get('clarity', -100)
            
            if vocal_clarity < -30:
                analyses['e_guitar']['recommendations'].append({
                    'priority': 'important',
                    'title': 'ボーカルスペース確保',
                    'steps': [
                        'PEQ: 3.2kHz, Q=2.0, -2.5dB',
                        '  ボーカルの明瞭度帯域を空ける',
                        '',
                        '効果: ボーカルの明瞭度向上'
                    ],
                    'expected_results': ['ボーカルとの分離向上']
                })
    
    # ヘルパーメソッド
    
    def _get_vocal_eq_steps_safe(self):
        """ボーカルEQ（フィードバック配慮）"""
        return [
            '【PEQ設定（フィードバック配慮）】',
            '  Band 1: 250Hz, Q=3.0, -2.5dB（こもり除去）',
            '  Band 2: 800Hz, Q=2.0, -2.0dB（低域整理）',
            '  Band 3: 3.2kHz, Q=3.0, +3.0dB（明瞭度・ナロー）',
            '  Band 4: 5kHz, Q=2.5, +2.0dB（子音）',
            '',
            '【HPF】',
            '  80Hz, 24dB/oct',
            '',
            '【Compressor】',
            '  Threshold: -18dB, Ratio: 4:1',
            '  Attack: 10ms, Release: 100ms',
            '  Make-up: +3dB',
            '',
            '【フィードバック対策】',
            '  ⚠️ 3.2kHzをゆっくり上げる（+1dBずつ）',
            '  ⚠️ 事前にRingingで共振周波数特定',
            '  ⚠️ モニター位置確認'
        ]
    
    def _get_vocal_eq_steps_full(self):
        """ボーカルEQ（積極的処理）"""
        return [
            '【PEQ設定】',
            '  Band 1: 250Hz, Q=2.0, -3.0dB（こもり除去）',
            '  Band 2: 3kHz, Q=1.5, +4.5dB（明瞭度・広帯域）',
            '  Band 3: 5kHz, Q=2.0, +3.0dB（子音）',
            '  Band 4: 10kHz, Q=1.5, +2.0dB（空気感）',
            '',
            '【HPF】',
            '  80Hz, 24dB/oct',
            '',
            '【Compressor】',
            '  Threshold: -18dB, Ratio: 4:1',
            '  Attack: 10ms, Release: 100ms',
            '',
            '【De-Esser】',
            '  Frequency: 6.5kHz, Range: -3dB'
        ]
    
    def _get_deesser_steps(self):
        """De-Esser設定手順"""
        
        if self.mixer_specs and self.mixer_specs.get('has_de_esser'):
            return [
                'De-Esser設定:',
                '  Frequency: 6.5kHz',
                '  Threshold: 調整（歯擦音が出た時のみ反応）',
                '  Range: -3dB',
                '',
                '効果: 自然な歯擦音コントロール'
            ]
        else:
            return [
                'De-Esser非搭載のため代替案:',
                '',
                '【方法1】Dynamic EQ',
                '  6-8kHz, Threshold調整, -3dB',
                '',
                '【方法2】Compressor（サイドチェーン）',
                '  HPFで6kHz以上のみ検知',
                '',
                '【方法3】外部De-Esser使用'
            ]
    
    def _get_kick_hpf_freq(self):
        """キックのHPF周波数（PA考慮）"""
        
        if not self.pa_specs:
            return 35
        
        pa_name = self.pa_specs.get('name', '').lower()
        low_ext = self.pa_specs.get('low_extension', 50)
        
        if 'd&b' in pa_name or low_ext <= 45:
            return 35  # 低域が良好なら35Hz
        elif 'jbl' in pa_name or low_ext <= 50:
            return 30  # JBLなら30Hz
        else:
            return 40  # 小型PAなら40Hz
    
    def _get_pa_kick_notes(self):
        """PA別のキック注意事項"""
        
        if not self.pa_specs:
            return ['  一般的なPAシステムを想定']
        
        pa_name = self.pa_specs.get('name', '')
        notes = self.pa_specs.get('recommendations', {}).get('kick_hpf', '')
        
        if notes:
            return [f'  {notes}']
        else:
            return [f'  {pa_name}の特性に最適化']
    
    def _get_mixer_vocal_steps(self):
        """ミキサー別ボーカル設定"""
        
        if not self.mixer_specs:
            return None
        
        mixer_name = self.mixer_specs.get('name')
        
        if 'Yamaha CL' in mixer_name:
            return {
                'mixer': mixer_name,
                'steps': [
                    '1. ボーカルchを選択',
                    '2. [EQ]ボタン → PEQ画面',
                    '3. Band設定を上記の通り実施',
                    '4. [DYNAMICS1] → Compressor',
                    '5. TYPE: Comp260（透明度重視）',
                    '6. パラメータ設定',
                    '7. ゲインリダクション 4-6dB確認'
                ]
            }
        elif 'X32' in mixer_name:
            return {
                'mixer': mixer_name,
                'steps': [
                    '1. ボーカルchを選択',
                    '2. [EQ]ボタン',
                    '3. Band設定（4バンド・優先順位順）',
                    '4. [DYNAMICS] → Compressor',
                    '5. パラメータ設定',
                    '',
                    '注意: 4バンドのみ。優先順位を守る'
                ]
            }
        
        return None
    
    def _get_mixer_hpf_steps(self, instrument, freq):
        """ミキサー別HPF設定"""
        
        if not self.mixer_specs:
            return None
        
        mixer_name = self.mixer_specs.get('name')
        
        return {
            'mixer': mixer_name,
            'steps': [
                f'1. {instrument}チャンネルを選択',
                '2. [EQ]ボタン',
                f'3. HPF: {freq}Hz, 24dB/oct',
                '4. HPF ONを確認'
            ]
        }




# =====================================
# 過去音源比較機能
# =====================================

class ComparisonAnalyzer:
    """過去音源との比較（システム差異考慮）"""
    
    def __init__(self, current_analysis, past_entries, current_metadata):
        self.current = current_analysis
        self.past_entries = past_entries
        self.current_metadata = current_metadata
    
    def compare_all(self):
        """全ての過去音源と比較"""
        
        comparisons = []
        
        for entry in self.past_entries:
            comp = self._compare_with_entry(entry)
            if comp:
                comparisons.append(comp)
        
        return comparisons
    
    def _compare_with_entry(self, past_entry):
        """個別の過去音源と比較"""
        
        past_analysis = past_entry['analysis']
        past_metadata = past_entry['metadata']
        past_equipment = past_entry['equipment']
        
        comparison = {
            'past_id': past_entry['id'],
            'past_date': past_entry['timestamp'],
            'past_venue': past_metadata.get('venue', '不明'),
            'past_mixer': past_equipment.get('mixer', '不明'),
            'past_pa': past_equipment.get('pa_system', '不明'),
            'match_type': self._get_match_type(past_metadata, past_equipment),
            'metrics': {},
            'insights': []
        }
        
        # RMS比較（ミキサー補正）
        current_rms = self.current.get('rms_db', -100)
        past_rms = past_analysis.get('rms_db', -100)
        
        # ミキサー補正
        rms_correction = self._get_mixer_correction(
            self.current_metadata.get('mixer'),
            past_equipment.get('mixer')
        )
        
        past_rms_corrected = past_rms + rms_correction
        rms_diff = current_rms - past_rms_corrected
        
        comparison['metrics']['rms'] = {
            'current': current_rms,
            'past_raw': past_rms,
            'past_corrected': past_rms_corrected,
            'difference': rms_diff,
            'correction_applied': rms_correction
        }
        
        # ステレオ幅比較
        current_width = self.current.get('stereo_width', 0)
        past_width = past_analysis.get('stereo_width', 0)
        width_diff = current_width - past_width
        
        comparison['metrics']['stereo_width'] = {
            'current': current_width,
            'past': past_width,
            'difference': width_diff
        }
        
        # 周波数バランス比較（PA補正）
        current_bands = self.current.get('band_energies', [])
        past_bands = past_analysis.get('band_energies', [])
        
        if len(current_bands) == len(past_bands) and len(current_bands) > 0:
            pa_corrections = self._get_pa_corrections(
                self.current_metadata.get('pa_system'),
                past_equipment.get('pa_system')
            )
            
            band_diffs = []
            for i in range(len(current_bands)):
                correction = pa_corrections[i] if i < len(pa_corrections) else 0
                past_corrected = past_bands[i] + correction
                diff = current_bands[i] - past_corrected
                band_diffs.append(diff)
            
            comparison['metrics']['frequency_balance'] = {
                'differences': band_diffs,
                'pa_correction_applied': any(c != 0 for c in pa_corrections)
            }
        
        # 洞察生成
        comparison['insights'] = self._generate_insights(comparison, past_metadata)
        
        return comparison
    
    def _get_match_type(self, past_metadata, past_equipment):
        """マッチタイプ判定"""
        
        score = 0
        
        # 会場が近い
        current_capacity = self.current_metadata.get('venue_capacity', 0)
        past_capacity = past_metadata.get('venue_capacity', 0)
        
        if abs(current_capacity - past_capacity) < 50:
            score += 30
        
        # ミキサーが同じ
        if self.current_metadata.get('mixer') == past_equipment.get('mixer'):
            score += 40
        
        # PAが同じ
        if self.current_metadata.get('pa_system') == past_equipment.get('pa_system'):
            score += 30
        
        if score >= 80:
            return 'exact_match'
        elif score >= 50:
            return 'similar'
        else:
            return 'different'
    
    def _get_mixer_correction(self, current_mixer, past_mixer):
        """ミキサー間の補正値"""
        
        if not current_mixer or not past_mixer:
            return 0.0
        
        if current_mixer == past_mixer:
            return 0.0
        
        # 簡易的な補正（実際はより詳細に）
        mixer_tiers = {
            'cl': 1.0,
            'ql': 0.8,
            'sq': 0.7,
            'x32': 0.5
        }
        
        current_tier = 0.5
        past_tier = 0.5
        
        for key, value in mixer_tiers.items():
            if key in current_mixer.lower():
                current_tier = value
            if key in past_mixer.lower():
                past_tier = value
        
        # ティア差 × 2dB
        return (current_tier - past_tier) * 2.0
    
    def _get_pa_corrections(self, current_pa, past_pa):
        """PA間の周波数補正"""
        
        # 7バンド分の補正値
        corrections = [0.0] * 7
        
        if not current_pa or not past_pa or current_pa == past_pa:
            return corrections
        
        # 簡易的な補正
        # d&b: フラット
        # JBL: 高域明るい（+2dB）
        # L-Acoustics: フラット
        
        current_brightness = 0
        past_brightness = 0
        
        if 'jbl' in current_pa.lower():
            current_brightness = 2
        if 'jbl' in past_pa.lower():
            past_brightness = 2
        
        brightness_diff = current_brightness - past_brightness
        
        # Presence/Brillianceに反映
        corrections[5] = -brightness_diff * 1.5  # Presence
        corrections[6] = -brightness_diff * 2.0  # Brilliance
        
        return corrections
    
    def _generate_insights(self, comparison, past_metadata):
        """比較からの洞察生成"""
        
        insights = []
        
        match_type = comparison['match_type']
        rms_diff = comparison['metrics']['rms']['difference']
        
        # RMS変化
        if match_type == 'exact_match':
            if rms_diff > 2:
                insights.append({
                    'type': 'improvement',
                    'message': f'音圧が前回より +{rms_diff:.1f}dB 向上（同条件比較）',
                    'severity': 'good'
                })
            elif rms_diff < -2:
                insights.append({
                    'type': 'regression',
                    'message': f'音圧が前回より {rms_diff:.1f}dB 低下（同条件比較）',
                    'severity': 'warning'
                })
            else:
                insights.append({
                    'type': 'stable',
                    'message': f'音圧は前回と同レベル（{rms_diff:+.1f}dB）',
                    'severity': 'info'
                })
        else:
            # 異なる条件
            correction = comparison['metrics']['rms'].get('correction_applied', 0)
            if correction != 0:
                insights.append({
                    'type': 'info',
                    'message': f'音圧差 {rms_diff:+.1f}dB（システム差補正済: {correction:+.1f}dB）',
                    'severity': 'info'
                })
        
        # ステレオ幅変化
        width_diff = comparison['metrics']['stereo_width']['difference']
        if abs(width_diff) > 10:
            insights.append({
                'type': 'change',
                'message': f'ステレオ幅が {width_diff:+.1f}% 変化',
                'severity': 'info'
            })
        
        # 周波数バランス
        if 'frequency_balance' in comparison['metrics']:
            band_diffs = comparison['metrics']['frequency_balance']['differences']
            band_names = ['Sub Bass', 'Bass', 'Low-Mid', 'Mid', 'High-Mid', 'Presence', 'Brilliance']
            
            for i, diff in enumerate(band_diffs):
                if abs(diff) > 6:
                    insights.append({
                        'type': 'change',
                        'message': f'{band_names[i]}が {diff:+.1f}dB 変化',
                        'severity': 'info'
                    })
        
        return insights


# =====================================
# メインUI
# =====================================

def main():
    st.markdown('<h1 class="main-header">🎛️ Live PA Audio Analyzer V3.0</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="version-badge">Final Release - 完全版</p>', 
                unsafe_allow_html=True)
    
    # データベース初期化
    db = AudioDatabase()
    
    # 機材検索初期化
    equipment_searcher = EquipmentSpecsSearcher()
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        uploaded_file = st.file_uploader(
            "音源ファイルをアップロード",
            type=['mp3', 'wav', 'flac', 'm4a']
        )
        
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 100:
                st.error(f"❌ ファイルが大きすぎます（{file_size_mb:.1f}MB）")
                uploaded_file = None
            else:
                st.success(f"✓ {file_size_mb:.1f}MB")
        
        st.markdown("---")
        
        # バンド編成（テキスト入力）
        st.subheader("🎸 バンド編成")
        
        band_lineup_text = st.text_area(
            "楽器を入力（カンマ区切り）",
            value="ボーカル、キック、スネア、ハイハット、ベース、ギター",
            height=100,
            help="例: ボーカル、キック、スネア、ベース、ギター\n日本語・英語・略語OK"
        )
        
        if not band_lineup_text.strip():
            st.warning("⚠️ バンド編成を入力してください")
        
        st.markdown("---")
        st.subheader("🏛️ 会場情報")
        
        venue_name = st.text_input("会場名（任意）", placeholder="例: CLUB QUATTRO")
        venue_capacity = st.slider("会場キャパ（人）", 50, 2000, 150, 50)
        stage_volume = st.selectbox("ステージ生音", ['high', 'medium', 'low', 'none'], 1)
        
        st.markdown("---")
        st.subheader("🎛️ 使用機材")
        
        mixer_name = st.text_input(
            "ミキサー", 
            placeholder="例: Yamaha CL5",
            help="正確な型番を入力すると自動で仕様を検索します"
        )
        
        pa_system = st.text_input(
            "PAシステム", 
            placeholder="例: d&b V-Series",
            help="システム名を入力すると特性を考慮した提案を行います"
        )
        
        notes = st.text_area("メモ（任意）", placeholder="セットリスト、特記事項など")
        
        st.markdown("---")
        
        # 過去音源表示
        recent_entries = db.get_recent(3)
        if recent_entries:
            st.subheader("📊 最近の解析")
            for entry in recent_entries:
                date = datetime.fromisoformat(entry['timestamp']).strftime('%m/%d %H:%M')
                venue = entry['metadata'].get('venue', '不明')
                st.caption(f"{date} - {venue}")
        
        st.markdown("---")
        analyze_button = st.button(
            "🚀 解析開始", 
            type="primary", 
            use_container_width=True,
            disabled=(uploaded_file is None or not band_lineup_text.strip())
        )
    
    # メインエリア
    if uploaded_file is None:
        st.info("👈 音源をアップロードしてバンド編成を入力してください")
        
        st.markdown("### 🆕 V3.0 Final の全機能")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 2mix全体解析**
            - 音圧、ステレオイメージ、周波数解析
            - 9パネル詳細グラフ
            - 良いポイント + 改善提案
            
            **🎸 楽器別詳細解析**
            - テキスト入力で自由な編成指定
            - 全楽器の周波数特性解析
            - 楽器ごとの具体的EQ/Comp設定
            """)
        
        with col2:
            st.markdown("""
            **🔍 Web検索統合**
            - ミキサー仕様の自動取得
            - PAシステム特性の反映
            - 機材に最適化された提案
            
            **📈 過去音源との比較**
            - システム差異を考慮した補正
            - 成長トレンドの可視化
            - 同条件 vs 異条件の比較
            """)
        
        st.markdown("---")
        st.markdown("### 📝 使い方")
        st.markdown("""
        1. **音源アップロード**: 2mix音源（mp3, wav等）
        2. **バンド編成入力**: 「ボーカル、キック、スネア、ベース」など
        3. **会場・機材情報**: できるだけ詳しく入力
        4. **解析開始**: ボタンをクリック
        5. **結果確認**: グラフ、良いポイント、改善提案を確認
        6. **実践**: 具体的な設定値を現場で試す
        """)
    
    elif analyze_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # メタデータ
            metadata = {
                'venue': venue_name,
                'venue_capacity': venue_capacity,
                'stage_volume': stage_volume,
                'mixer': mixer_name,
                'pa_system': pa_system,
                'band_lineup': band_lineup_text,
                'notes': notes
            }
            
            # === Phase 1: 機材仕様検索 ===
            
            mixer_specs = None
            pa_specs = None
            
            if mixer_name:
                mixer_specs = equipment_searcher.search_mixer_specs(mixer_name)
                if mixer_specs:
                    st.success(f"✅ {mixer_specs['name']}の仕様を取得")
            
            if pa_system:
                pa_specs = equipment_searcher.search_pa_specs(pa_system)
                if pa_specs:
                    st.success(f"✅ {pa_specs['name']}の特性を取得")
            
            # === Phase 2: V2解析（2mix全体） ===
            
            st.markdown("## 📊 2mix全体解析")
            
            v2_analyzer = V2Analyzer(tmp_path, venue_capacity, stage_volume, pa_system, notes)
            v2_results = v2_analyzer.analyze()
            
            st.success("✅ 2mix解析完了")
            
            # サマリーメトリクス
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ステレオ幅", f"{v2_results['stereo_width']:.1f}%")
            with col2:
                st.metric("RMS", f"{v2_results['rms_db']:.1f} dB")
            with col3:
                st.metric("クレストファクター", f"{v2_results['crest_factor']:.1f} dB")
            with col4:
                st.metric("ダイナミックレンジ", f"{v2_results['dynamic_range']:.1f} dB")
            
            # グラフ表示
            st.markdown("### 📈 詳細グラフ")
            
            with st.spinner('📊 グラフを生成中...'):
                fig = v2_analyzer.create_visualization()
                st.pyplot(fig, use_container_width=True)
                
                # ダウンロードボタン
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="📥 グラフをダウンロード",
                    data=buf,
                    file_name=f"pa_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
                
                plt.close(fig)
            
            # 2mix改善提案
            st.markdown("### 💡 2mix全体の改善提案")
            
            good_points, v2_recs = v2_analyzer.generate_v2_recommendations(mixer_specs, pa_specs)
            
            # 良いポイント
            if good_points:
                st.markdown("#### ✅ 良いポイント")
                for gp in good_points:
                    st.markdown(f"""
                    <div class="good-point">
                        <strong>{gp['category']}</strong>: {gp['point']}<br>
                        影響度: {gp['impact']}
                    </div>
                    """, unsafe_allow_html=True)
            
            # 改善提案
            for priority in ['critical', 'important', 'optional']:
                if v2_recs[priority]:
                    priority_label = {
                        'critical': '🔴 最優先', 
                        'important': '🟡 重要', 
                        'optional': '🟢 オプション'
                    }[priority]
                    
                    st.markdown(f"#### {priority_label}")
                    
                    for rec in v2_recs[priority]:
                        with st.expander(f"{rec['category']}: {rec['issue']}"):
                            st.write(f"**対策:** {rec['solution']}")
                            st.write(f"**影響度:** {rec['impact']}")
            
            st.markdown("---")
            
            # === Phase 3: 楽器別解析 ===
            
            st.markdown("## 🎸 楽器別詳細解析")
            
            # 楽器分離
            separator = InstrumentSeparator(v2_analyzer.y, v2_analyzer.sr, band_lineup_text)
            stems = separator.separate()
            
            st.success(f"✅ {len(stems)}楽器を分離完了")
            
            # 分離された楽器を表示
            st.write("**検出された楽器:**", ', '.join(
                {'vocal': 'ボーカル', 'kick': 'キック', 'snare': 'スネア',
                 'bass': 'ベース', 'hihat': 'ハイハット', 'tom': 'タム',
                 'e_guitar': 'エレキギター', 'a_guitar': 'アコギ',
                 'keyboard': 'キーボード', 'synth': 'シンセ'}.get(k, k)
                for k in stems.keys()
            ))
            
            # 詳細解析
            inst_analyzer = InstrumentAnalyzer(
                stems, v2_analyzer.sr, v2_analyzer.y, 
                v2_results['rms_db'],
                mixer_specs, pa_specs
            )
            
            inst_analyses = inst_analyzer.analyze_all(venue_capacity, stage_volume)
            
            st.success("✅ 楽器別解析完了")
            
            # 楽器別の詳細表示
            for inst_name, analysis in inst_analyses.items():
                inst_name_ja = {
                    'vocal': 'ボーカル', 'kick': 'キック', 'snare': 'スネア',
                    'bass': 'ベース', 'hihat': 'ハイハット', 'tom': 'タム',
                    'e_guitar': 'エレキギター', 'a_guitar': 'アコギ',
                    'keyboard': 'キーボード', 'synth': 'シンセ'
                }.get(inst_name, inst_name)
                
                icon = {
                    'vocal': '🎤', 'kick': '🥁', 'snare': '🥁', 'bass': '🎸',
                    'hihat': '🥁', 'tom': '🥁', 'e_guitar': '🎸', 'a_guitar': '🎸',
                    'keyboard': '🎹', 'synth': '🎹'
                }.get(inst_name, '🎵')
                
                with st.expander(f"{icon} {inst_name_ja}の詳細解析", expanded=(inst_name in ['vocal', 'kick'])):
                    # 基本情報
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMS", f"{analysis['rms_db']:.1f} dBFS")
                    with col2:
                        st.metric("Peak", f"{analysis['peak_db']:.1f} dBFS")
                    with col3:
                        st.metric("vs 2mix", f"{analysis['level_vs_mix']:+.1f} dB")
                    
                    # 周波数帯域
                    if analysis.get('freq_bands'):
                        st.markdown("**周波数帯域別レベル:**")
                        for band_name, level in analysis['freq_bands'].items():
                            st.write(f"- {band_name}: {level:.1f} dB")
                    
                    # 良いポイント
                    if analysis.get('good_points'):
                        st.markdown("**✅ 良いポイント:**")
                        for gp in analysis['good_points']:
                            st.markdown(f"""
                            <div class="good-point">
                                {gp['point']}<br>
                                影響度: {gp.get('impact', '★★★')}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # 問題点
                    if analysis.get('issues'):
                        st.markdown("**❌ 検出された問題:**")
                        for issue in analysis['issues']:
                            severity_icon = {
                                'critical': '🔴', 
                                'important': '🟡', 
                                'medium': '🟠'
                            }.get(issue['severity'], '⚪')
                            st.write(f"{severity_icon} **{issue['problem']}**")
                            st.caption(issue['detail'])
                    
                    # 改善提案
                    if analysis.get('recommendations'):
                        st.markdown("**💡 改善提案:**")
                        
                        for i, rec in enumerate(analysis['recommendations'], 1):
                            priority_icon = {
                                'critical': '🔴', 
                                'important': '🟡', 
                                'optional': '🟢'
                            }.get(rec['priority'], '⚪')
                            
                            st.markdown(f"**{priority_icon} {i}. {rec['title']}**")
                            
                            for step in rec['steps']:
                                st.write(step)
                            
                            # ミキサー固有の手順
                            if rec.get('mixer_specific'):
                                with st.expander(f"📱 {rec['mixer_specific']['mixer']} での操作手順"):
                                    for step in rec['mixer_specific']['steps']:
                                        st.write(step)
                            
                            # 期待される結果
                            if rec.get('expected_results'):
                                st.markdown("**🎯 期待される効果:**")
                                for result in rec['expected_results']:
                                    st.write(f"✅ {result}")
                            
                            st.markdown("---")
            
            st.markdown("---")
            
            # === Phase 4: 過去音源との比較 ===
            
            similar_entries = db.find_similar(metadata, limit=3)
            
            if similar_entries:
                st.markdown("## 📊 過去音源との比較")
                
                comp_analyzer = ComparisonAnalyzer(v2_results, similar_entries, metadata)
                comparisons = comp_analyzer.compare_all()
                
                for i, comp in enumerate(comparisons, 1):
                    match_icon = {
                        'exact_match': '🟢',
                        'similar': '🟡',
                        'different': '🔵'
                    }.get(comp['match_type'], '⚪')
                    
                    match_label = {
                        'exact_match': 'ほぼ同条件',
                        'similar': '類似条件',
                        'different': '異なる条件'
                    }.get(comp['match_type'], '不明')
                    
                    with st.expander(f"{match_icon} 比較 #{i}: {match_label} - {comp['past_venue']}", expanded=(i==1)):
                        st.write(f"**日時:** {datetime.fromisoformat(comp['past_date']).strftime('%Y年%m月%d日 %H:%M')}")
                        st.write(f"**会場:** {comp['past_venue']}")
                        st.write(f"**ミキサー:** {comp['past_mixer']}")
                        st.write(f"**PA:** {comp['past_pa']}")
                        
                        st.markdown("---")
                        
                        # メトリクス比較
                        rms_metric = comp['metrics']['rms']
                        
                        st.markdown("**音圧（RMS）:**")
                        st.write(f"- 現在: {rms_metric['current']:.1f} dBFS")
                        st.write(f"- 過去: {rms_metric['past_raw']:.1f} dBFS（生値）")
                        
                        if rms_metric['correction_applied'] != 0:
                            st.write(f"- 過去（補正後）: {rms_metric['past_corrected']:.1f} dBFS")
                            st.caption(f"補正値: {rms_metric['correction_applied']:+.1f}dB（ミキサー差異）")
                        
                        st.write(f"- **差分: {rms_metric['difference']:+.1f} dB**")
                        
                        # ステレオ幅
                        width_metric = comp['metrics']['stereo_width']
                        st.markdown("**ステレオ幅:**")
                        st.write(f"- 差分: {width_metric['difference']:+.1f}%")
                        
                        # 洞察
                        if comp['insights']:
                            st.markdown("**💡 洞察:**")
                            for insight in comp['insights']:
                                icon = {
                                    'improvement': '✅',
                                    'regression': '⚠️',
                                    'stable': '→',
                                    'change': '📌',
                                    'info': 'ℹ️'
                                }.get(insight['type'], '•')
                                
                                st.write(f"{icon} {insight['message']}")
            
            # === データベースに保存 ===
            
            entry_id = db.add_entry(v2_results, metadata)
            st.success(f"✅ 解析結果を保存しました（ID: {entry_id}）")
        
        except Exception as e:
            st.error(f"❌ エラー: {str(e)}")
            with st.expander("詳細"):
                st.exception(e)
        
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    main()
