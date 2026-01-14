#!/usr/bin/env python3
"""
Web-based visualization tool for classifier comparison results.

Displays side-by-side predictions from two classifiers (e.g., multimodal vs audio-only)
with filtering to focus on disagreements and cases where one model outperforms the other.

Run with:
    python visualize_comparison.py <comparison.jsonl> --dataset-path /path/to/dataset

The dataset path should contain video/ and audio/ subdirectories with the media files.

Then open http://localhost:5051 in your browser.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

from flask import (
    Flask,
    render_template_string,
    request,
    jsonify,
    send_from_directory,
)

app = Flask(__name__)

# Global state
COMPARISON_PATH = None
DATASET_PATH = None  # Local dataset directory with video/ and audio/ subdirs
ENTRIES = []
VIDEO_GROUPS = {}  # prefix -> list of entry indices


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classifier Comparison</title>
    <style>
        /* Dark theme (default) */
        :root {
            --bg-primary: #111113;
            --bg-secondary: #18181b;
            --bg-tertiary: #202024;
            --bg-card: #1c1c1f;
            --border-color: #2e2e33;
            --border-highlight: #3f3f46;
            --text-primary: #f4f4f5;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --correct: #22c55e;
            --correct-dim: rgba(34, 197, 94, 0.12);
            --incorrect: #ef4444;
            --incorrect-dim: rgba(239, 68, 68, 0.12);
            --model-a: #52525b;
            --model-b: #71717a;
            --highlight: rgba(255, 255, 255, 0.06);
        }

        /* Light theme */
        [data-theme="light"] {
            --bg-primary: #fafafa;
            --bg-secondary: #f4f4f5;
            --bg-tertiary: #e4e4e7;
            --bg-card: #ffffff;
            --border-color: #d4d4d8;
            --border-highlight: #a1a1aa;
            --text-primary: #18181b;
            --text-secondary: #52525b;
            --text-muted: #71717a;
            --correct: #16a34a;
            --correct-dim: rgba(22, 163, 74, 0.1);
            --incorrect: #dc2626;
            --incorrect-dim: rgba(220, 38, 38, 0.1);
            --model-a: #71717a;
            --model-b: #a1a1aa;
            --highlight: rgba(0, 0, 0, 0.04);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            transition: background-color 0.2s ease, color 0.2s ease;
        }

        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 28px;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1800px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .header h1 {
            font-size: 18px;
            font-weight: 600;
            letter-spacing: -0.3px;
            color: var(--text-primary);
        }

        .theme-toggle {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            width: 36px;
            height: 36px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: all 0.2s ease;
        }

        .theme-toggle:hover {
            background: var(--border-color);
            color: var(--text-primary);
        }

        .stats-bar {
            display: flex;
            gap: 24px;
            align-items: center;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }

        .stat-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }

        .nav-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .nav-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 18px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .nav-btn:hover:not(:disabled) {
            background: var(--border-color);
            border-color: var(--border-highlight);
            transform: translateY(-1px);
        }

        .nav-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .video-select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 14px;
            max-width: 350px;
            cursor: pointer;
        }

        .main-content {
            max-width: 1800px;
            margin: 0 auto;
            padding: 28px;
        }

        .filter-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            padding: 16px 20px;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        .filter-btn {
            padding: 8px 14px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .filter-btn:hover {
            color: var(--text-primary);
            border-color: var(--border-highlight);
        }

        .filter-btn.active {
            background: var(--text-primary);
            border-color: var(--text-primary);
            color: var(--bg-primary);
        }

        .filter-count {
            opacity: 0.7;
            font-size: 12px;
            font-variant-numeric: tabular-nums;
        }

        .video-group-title {
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 20px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .group-stats {
            display: flex;
            gap: 16px;
            font-size: 13px;
            color: var(--text-muted);
        }

        .segments-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
            gap: 20px;
        }

        .segment-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.2s ease;
        }

        .segment-card:hover {
            border-color: var(--border-highlight);
        }

        .segment-card.mm-better {
            border-left: 3px solid var(--text-muted);
        }

        .segment-card.ao-better {
            border-left: 3px solid var(--text-muted);
            border-left-style: dashed;
        }

        .segment-card.both-wrong {
            border-left: 3px solid var(--incorrect);
        }

        .segment-card.both-correct {
            border-left: 3px solid var(--correct);
        }

        .video-container {
            position: relative;
            background: #000;
        }

        .video-container video {
            width: 100%;
            display: block;
            max-height: 280px;
            object-fit: contain;
        }

        .ground-truth-badge {
            position: absolute;
            top: 10px;
            left: 10px;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            backdrop-filter: blur(8px);
        }

        .ground-truth-badge.eot {
            background: rgba(255, 255, 255, 0.9);
            color: #000;
        }

        .ground-truth-badge.not-eot {
            background: rgba(0, 0, 0, 0.75);
            color: #fff;
        }

        .segment-info {
            padding: 20px;
        }

        .segment-name {
            font-size: 12px;
            font-family: 'SF Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
            color: var(--text-muted);
            margin-bottom: 16px;
            word-break: break-all;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }

        .predictions-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .prediction-box {
            padding: 14px;
            border-radius: 8px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
        }

        .prediction-box.multimodal {
            border-top: 2px solid var(--text-primary);
        }

        .prediction-box.audio-only {
            border-top: 2px dashed var(--text-muted);
        }

        .prediction-box.correct {
            background: var(--correct-dim);
        }

        .prediction-box.incorrect {
            background: var(--incorrect-dim);
        }

        .pred-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .pred-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            font-weight: 600;
        }

        .pred-result {
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 600;
        }

        .pred-result.correct {
            background: var(--correct);
            color: #fff;
        }

        .pred-result.incorrect {
            background: var(--incorrect);
            color: #fff;
        }

        .prob-bar-container {
            margin-bottom: 8px;
        }

        .prob-bar {
            height: 6px;
            background: var(--bg-primary);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }

        .prob-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.2s ease;
            background: var(--text-muted);
        }

        .threshold-line {
            position: absolute;
            top: -2px;
            bottom: -2px;
            width: 2px;
            background: var(--text-secondary);
        }

        .prob-value {
            font-size: 22px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            color: var(--text-primary);
        }

        .pred-label {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .comparison-summary {
            margin-top: 14px;
            padding: 10px 14px;
            background: var(--highlight);
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .summary-badge {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .summary-badge.both-correct {
            color: var(--correct);
        }

        .summary-badge.both-wrong {
            color: var(--incorrect);
        }

        .prob-diff {
            font-size: 12px;
            color: var(--text-muted);
            font-variant-numeric: tabular-nums;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }

        .empty-state-icon {
            font-size: 32px;
            margin-bottom: 12px;
            opacity: 0.5;
        }

        kbd {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 11px;
            font-family: inherit;
        }

        @media (max-width: 1100px) {
            .segments-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-bar {
                display: none;
            }
        }

        /* Smooth transitions for theme switch */
        .header, .filter-bar, .video-group-title, .segment-card, 
        .prediction-box, .nav-btn, .filter-btn, .video-select {
            transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="header-left">
                <h1>Classifier Comparison</h1>
                <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" title="Toggle theme">
                    <span id="themeIcon">☀</span>
                </button>
            </div>
            
            <div class="stats-bar" id="statsBar">
                <div class="stat-item">
                    <div class="stat-value" id="mmAccStat">-</div>
                    <div class="stat-label">Multimodal</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="aoAccStat">-</div>
                    <div class="stat-label">Audio-Only</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="agreeStat">-</div>
                    <div class="stat-label">Agree</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="differStat">-</div>
                    <div class="stat-label">Disagree</div>
                </div>
            </div>

            <div class="nav-controls">
                <button class="nav-btn" id="prevBtn" onclick="navigateVideo(-1)">← Prev</button>
                <select class="video-select" id="videoSelect" onchange="loadVideoGroup(this.value)"></select>
                <button class="nav-btn" id="nextBtn" onclick="navigateVideo(1)">Next →</button>
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="filter-bar">
            <button class="filter-btn active" data-filter="all" onclick="setFilter('all')">
                All <span class="filter-count" id="countAll">0</span>
            </button>
            <button class="filter-btn" data-filter="disagree" onclick="setFilter('disagree')">
                Disagreements <span class="filter-count" id="countDisagree">0</span>
            </button>
            <button class="filter-btn" data-filter="mm_better" onclick="setFilter('mm_better')">
                MM Better <span class="filter-count" id="countMmBetter">0</span>
            </button>
            <button class="filter-btn" data-filter="ao_better" onclick="setFilter('ao_better')">
                Audio Better <span class="filter-count" id="countAoBetter">0</span>
            </button>
            <button class="filter-btn" data-filter="both_wrong" onclick="setFilter('both_wrong')">
                Both Wrong <span class="filter-count" id="countBothWrong">0</span>
            </button>
            <button class="filter-btn" data-filter="both_correct" onclick="setFilter('both_correct')">
                Both Correct <span class="filter-count" id="countBothCorrect">0</span>
            </button>
        </div>

        <div class="video-group-title" id="groupTitle">
            <span>Loading...</span>
            <div class="group-stats" id="groupStats"></div>
        </div>
        
        <div class="segments-grid" id="segmentsGrid">
            <div class="empty-state">
                <div class="empty-state-icon">—</div>
                <p>Loading comparison data...</p>
            </div>
        </div>
    </div>

    <script>
        let videoGroups = {};
        let entries = [];
        let currentGroupIndex = 0;
        let groupKeys = [];
        let currentFilter = 'all';
        let globalStats = {};

        // Theme handling
        function initTheme() {
            const saved = localStorage.getItem('theme');
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            const theme = saved || (prefersDark ? 'dark' : 'light');
            setTheme(theme);
        }

        function setTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            document.getElementById('themeIcon').textContent = theme === 'dark' ? '☀' : '☾';
        }

        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            setTheme(current === 'dark' ? 'light' : 'dark');
        }

        initTheme();

        async function init() {
            const response = await fetch('/api/data');
            const data = await response.json();
            entries = data.entries;
            videoGroups = data.video_groups;
            groupKeys = Object.keys(videoGroups).sort();
            globalStats = data.stats;
            
            // Update global stats
            document.getElementById('mmAccStat').textContent = globalStats.mm_accuracy + '%';
            document.getElementById('aoAccStat').textContent = globalStats.ao_accuracy + '%';
            document.getElementById('agreeStat').textContent = globalStats.agrees_pct + '%';
            document.getElementById('differStat').textContent = globalStats.disagrees_pct + '%';
            
            // Update filter counts
            document.getElementById('countAll').textContent = entries.length;
            document.getElementById('countDisagree').textContent = entries.filter(e => !e.agrees).length;
            document.getElementById('countMmBetter').textContent = entries.filter(e => e.mm_better).length;
            document.getElementById('countAoBetter').textContent = entries.filter(e => e.ao_better).length;
            document.getElementById('countBothWrong').textContent = entries.filter(e => e.both_wrong).length;
            document.getElementById('countBothCorrect').textContent = entries.filter(e => e.both_correct).length;
            
            // Populate video selector
            const select = document.getElementById('videoSelect');
            select.innerHTML = groupKeys.map((key, idx) => {
                const group = videoGroups[key];
                const disagreeCount = group.filter(i => !entries[i].agrees).length;
                const marker = disagreeCount > 0 ? `[${disagreeCount}]` : '';
                return `<option value="${idx}">${key} ${marker}</option>`;
            }).join('');
            
            loadVideoGroup(0);
        }

        function setFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.filter === filter);
            });
            renderSegments();
        }

        function loadVideoGroup(index) {
            currentGroupIndex = parseInt(index);
            document.getElementById('videoSelect').value = currentGroupIndex;
            document.getElementById('prevBtn').disabled = currentGroupIndex === 0;
            document.getElementById('nextBtn').disabled = currentGroupIndex === groupKeys.length - 1;
            
            const groupKey = groupKeys[currentGroupIndex];
            const segmentIndices = videoGroups[groupKey];
            
            // Compute group stats
            const groupEntries = segmentIndices.map(i => entries[i]);
            const disagree = groupEntries.filter(e => !e.agrees).length;
            const mmBetter = groupEntries.filter(e => e.mm_better).length;
            const aoBetter = groupEntries.filter(e => e.ao_better).length;
            
            document.getElementById('groupTitle').innerHTML = `
                <span>${groupKey}</span>
                <div class="group-stats">
                    <span>${segmentIndices.length} total</span>
                    <span>${disagree} differ</span>
                    <span>${mmBetter} MM better</span>
                    <span>${aoBetter} audio better</span>
                </div>
            `;
            
            renderSegments();
        }

        function renderSegments() {
            const groupKey = groupKeys[currentGroupIndex];
            const segmentIndices = videoGroups[groupKey];
            
            let filteredIndices = segmentIndices;
            if (currentFilter === 'disagree') {
                filteredIndices = segmentIndices.filter(i => !entries[i].agrees);
            } else if (currentFilter === 'mm_better') {
                filteredIndices = segmentIndices.filter(i => entries[i].mm_better);
            } else if (currentFilter === 'ao_better') {
                filteredIndices = segmentIndices.filter(i => entries[i].ao_better);
            } else if (currentFilter === 'both_wrong') {
                filteredIndices = segmentIndices.filter(i => entries[i].both_wrong);
            } else if (currentFilter === 'both_correct') {
                filteredIndices = segmentIndices.filter(i => entries[i].both_correct);
            }

            if (filteredIndices.length === 0) {
                document.getElementById('segmentsGrid').innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">—</div>
                        <p>No ${currentFilter.replace('_', ' ')} segments in this video group</p>
                    </div>
                `;
                return;
            }

            document.getElementById('segmentsGrid').innerHTML = filteredIndices.map(idx => {
                const e = entries[idx];
                const mm = e.multimodal;
                const ao = e.audio_only;
                
                let cardClass = '';
                if (e.mm_better) cardClass = 'mm-better';
                else if (e.ao_better) cardClass = 'ao-better';
                else if (e.both_wrong) cardClass = 'both-wrong';
                else if (e.both_correct) cardClass = 'both-correct';
                
                const segmentName = e.id.split('/').pop();
                const truthLabel = e.true_label === 1 ? 'End of Turn' : 'Not End of Turn';
                const truthClass = e.true_label === 1 ? 'eot' : 'not-eot';
                
                // Summary badge
                let summaryClass = 'agree';
                let summaryText = 'Agree';
                if (e.mm_better) { summaryClass = 'mm-better'; summaryText = 'Multimodal correct'; }
                else if (e.ao_better) { summaryClass = 'ao-better'; summaryText = 'Audio-only correct'; }
                else if (e.both_wrong) { summaryClass = 'both-wrong'; summaryText = 'Both wrong'; }
                else if (e.both_correct) { summaryClass = 'both-correct'; summaryText = 'Both correct'; }
                
                const probDiff = Math.abs(mm.prob - ao.prob).toFixed(3);
                const threshold = e.threshold || 0.5;
                const thresholdPct = threshold * 100;
                
                return `
                    <div class="segment-card ${cardClass}" id="card-${idx}">
                        <div class="video-container">
                            <video controls preload="metadata">
                                <source src="/video?path=${encodeURIComponent(e.video_path)}" type="video/mp4">
                            </video>
                            <div class="ground-truth-badge ${truthClass}">${truthLabel}</div>
                        </div>
                        <div class="segment-info">
                            <div class="segment-name">${segmentName}</div>
                            
                            <div class="predictions-container">
                                <div class="prediction-box multimodal ${mm.correct ? 'correct' : 'incorrect'}">
                                    <div class="pred-header">
                                        <span class="pred-title">Multimodal</span>
                                        <span class="pred-result ${mm.correct ? 'correct' : 'incorrect'}">
                                            ${mm.correct ? '✓' : '✗'}
                                        </span>
                                    </div>
                                    <div class="prob-bar-container">
                                        <div class="prob-bar">
                                            <div class="prob-fill" style="width: ${mm.prob * 100}%"></div>
                                            <div class="threshold-line" style="left: ${thresholdPct}%"></div>
                                        </div>
                                    </div>
                                    <div class="prob-value">${(mm.prob * 100).toFixed(1)}%</div>
                                    <div class="pred-label">${mm.pred === 1 ? 'End of Turn' : 'Continue'}</div>
                                </div>
                                
                                <div class="prediction-box audio-only ${ao.correct ? 'correct' : 'incorrect'}">
                                    <div class="pred-header">
                                        <span class="pred-title">Audio-Only</span>
                                        <span class="pred-result ${ao.correct ? 'correct' : 'incorrect'}">
                                            ${ao.correct ? '✓' : '✗'}
                                        </span>
                                    </div>
                                    <div class="prob-bar-container">
                                        <div class="prob-bar">
                                            <div class="prob-fill" style="width: ${ao.prob * 100}%"></div>
                                            <div class="threshold-line" style="left: ${thresholdPct}%"></div>
                                        </div>
                                    </div>
                                    <div class="prob-value">${(ao.prob * 100).toFixed(1)}%</div>
                                    <div class="pred-label">${ao.pred === 1 ? 'End of Turn' : 'Continue'}</div>
                                </div>
                            </div>
                            
                            <div class="comparison-summary">
                                <span class="summary-badge ${summaryClass}">${summaryText}</span>
                                <span class="prob-diff">Δ ${probDiff}</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function navigateVideo(delta) {
            const newIndex = currentGroupIndex + delta;
            if (newIndex >= 0 && newIndex < groupKeys.length) {
                loadVideoGroup(newIndex);
            }
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            
            if (e.key === 'ArrowLeft') {
                navigateVideo(-1);
            } else if (e.key === 'ArrowRight') {
                navigateVideo(1);
            } else if (e.key === '1') {
                setFilter('all');
            } else if (e.key === '2') {
                setFilter('disagree');
            } else if (e.key === '3') {
                setFilter('mm_better');
            } else if (e.key === '4') {
                setFilter('ao_better');
            }
        });

        init();
    </script>
</body>
</html>
"""


def load_comparison_data():
    """Load comparison data from JSONL file."""
    global ENTRIES
    ENTRIES = []

    if not COMPARISON_PATH.exists():
        return

    with open(COMPARISON_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ENTRIES.append(json.loads(line))
                except json.JSONDecodeError:
                    continue


def group_by_video():
    """Group entries by original video (prefix before '_segment_')."""
    global VIDEO_GROUPS
    VIDEO_GROUPS = defaultdict(list)

    for idx, entry in enumerate(ENTRIES):
        entry_id = entry.get("id", "")
        filename = os.path.basename(entry_id)

        # Extract prefix before '_segment_'
        if "_segment_" in filename:
            prefix = filename.split("_segment_")[0]
        else:
            prefix = filename

        VIDEO_GROUPS[prefix].append(idx)


def compute_stats():
    """Compute summary statistics."""
    total = len(ENTRIES)
    if total == 0:
        return {}

    return {
        "total_records": total,
        "agrees": sum(1 for e in ENTRIES if e.get("agrees")),
        "agrees_pct": round(
            100 * sum(1 for e in ENTRIES if e.get("agrees")) / total, 1
        ),
        "disagrees": sum(1 for e in ENTRIES if not e.get("agrees")),
        "disagrees_pct": round(
            100 * sum(1 for e in ENTRIES if not e.get("agrees")) / total, 1
        ),
        "mm_better": sum(1 for e in ENTRIES if e.get("mm_better")),
        "ao_better": sum(1 for e in ENTRIES if e.get("ao_better")),
        "both_correct": sum(1 for e in ENTRIES if e.get("both_correct")),
        "both_wrong": sum(1 for e in ENTRIES if e.get("both_wrong")),
        "mm_accuracy": round(
            100
            * sum(1 for e in ENTRIES if e.get("multimodal", {}).get("correct"))
            / total,
            1,
        ),
        "ao_accuracy": round(
            100
            * sum(1 for e in ENTRIES if e.get("audio_only", {}).get("correct"))
            / total,
            1,
        ),
    }


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/data")
def get_data():
    load_comparison_data()
    group_by_video()
    stats = compute_stats()
    return jsonify(
        {
            "entries": ENTRIES,
            "video_groups": dict(VIDEO_GROUPS),
            "stats": stats,
        }
    )


@app.route("/video")
def serve_video():
    video_path = request.args.get("path", "")
    if not video_path:
        return "No path specified", 400

    # Extract just the filename from whatever path format
    filename = os.path.basename(video_path)

    # If dataset path is specified, serve from its video/ subdirectory
    if DATASET_PATH:
        video_dir = DATASET_PATH / "video"
        return send_from_directory(str(video_dir), filename)

    # Otherwise try the original path
    if os.path.isabs(video_path):
        directory = os.path.dirname(video_path)
    else:
        directory = os.path.dirname(video_path) or "."

    return send_from_directory(directory, filename)


@app.route("/audio")
def serve_audio():
    audio_path = request.args.get("path", "")
    if not audio_path:
        return "No path specified", 400

    # Extract just the filename from whatever path format
    filename = os.path.basename(audio_path)

    # If dataset path is specified, serve from its audio/ subdirectory
    if DATASET_PATH:
        audio_dir = DATASET_PATH / "audio"
        return send_from_directory(str(audio_dir), filename)

    # Otherwise try the original path
    if os.path.isabs(audio_path):
        directory = os.path.dirname(audio_path)
    else:
        directory = os.path.dirname(audio_path) or "."

    return send_from_directory(directory, filename)


def main():
    global COMPARISON_PATH, DATASET_PATH

    parser = argparse.ArgumentParser(
        description="Web-based visualization for classifier comparison."
    )
    parser.add_argument(
        "comparison_file",
        type=str,
        help="Path to comparison JSONL file (from compare_classifiers.py)",
    )
    parser.add_argument(
        "--dataset-path",
        "-d",
        type=str,
        default=None,
        help="Local dataset directory containing video/ and audio/ subdirectories",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5051,
        help="Port to run the server on (default: 5051)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    COMPARISON_PATH = Path(args.comparison_file).resolve()
    DATASET_PATH = (
        Path(args.dataset_path).resolve() if args.dataset_path else None
    )

    if not COMPARISON_PATH.exists():
        print(f"❌ Comparison file not found: {COMPARISON_PATH}")
        sys.exit(1)

    if DATASET_PATH:
        if not DATASET_PATH.exists():
            print(f"❌ Dataset path not found: {DATASET_PATH}")
            sys.exit(1)
        video_dir = DATASET_PATH / "video"
        audio_dir = DATASET_PATH / "audio"
        if not video_dir.exists():
            print(
                f"⚠️  Warning: video/ subdirectory not found in {DATASET_PATH}"
            )
        if not audio_dir.exists():
            print(
                f"⚠️  Warning: audio/ subdirectory not found in {DATASET_PATH}"
            )

    load_comparison_data()
    group_by_video()
    stats = compute_stats()

    print(f"📂 Comparison file: {COMPARISON_PATH}")
    if DATASET_PATH:
        print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"📋 Total entries: {len(ENTRIES)}")
    print(f"🎬 Video groups: {len(VIDEO_GROUPS)}")
    print()
    print(f"📊 Multimodal accuracy: {stats.get('mm_accuracy', 0)}%")
    print(f"📊 Audio-only accuracy: {stats.get('ao_accuracy', 0)}%")
    print(f"🤝 Classifiers agree: {stats.get('agrees_pct', 0)}%")
    print(f"⚡ Disagreements: {stats.get('disagrees', 0)}")
    print()
    print(f"🌐 Open http://{args.host}:{args.port} in your browser\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
