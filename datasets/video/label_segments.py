#!/usr/bin/env python3
"""
Web-based labeling tool for video segments.

Run with:
    python label_segments.py <dataset_dir>

Then open http://localhost:5050 in your browser.
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
DATASET_DIR = None
METADATA_PATH = None
ENTRIES = []
VIDEO_GROUPS = {}  # prefix -> list of entry indices


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Segment Labeler</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-green: #238636;
            --accent-green-hover: #2ea043;
            --accent-red: #da3633;
            --accent-red-hover: #f85149;
            --accent-blue: #1f6feb;
            --accent-yellow: #d29922;
            --accent-purple: #8957e5;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            min-height: 100vh;
        }

        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 24px;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }

        .header h1 {
            font-size: 20px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .progress-bar {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .progress-track {
            width: 200px;
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--accent-green);
            transition: width 0.3s ease;
        }

        .progress-text {
            font-size: 14px;
            color: var(--text-secondary);
        }

        .nav-controls {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .nav-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }

        .nav-btn:hover:not(:disabled) {
            background: var(--border-color);
        }

        .nav-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .video-select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            max-width: 300px;
        }

        .main-content {
            max-width: 1600px;
            margin: 0 auto;
            padding: 24px;
        }

        .video-group-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .segments-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }

        .segment-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            transition: border-color 0.2s;
        }

        .segment-card.labeled {
            border-color: var(--accent-green);
        }

        .segment-card.partial {
            border-color: var(--accent-yellow);
        }

        .video-container {
            position: relative;
            background: #000;
        }

        .video-container video {
            width: 100%;
            display: block;
        }

        .segment-info {
            padding: 16px;
        }

        .segment-name {
            font-size: 13px;
            font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace;
            color: var(--text-secondary);
            margin-bottom: 12px;
            word-break: break-all;
        }

        .label-group {
            margin-bottom: 16px;
        }

        .label-title {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .endpoint-buttons {
            display: flex;
            gap: 8px;
        }

        .endpoint-btn {
            flex: 1;
            padding: 10px 16px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }

        .endpoint-btn:hover {
            background: var(--border-color);
        }

        .endpoint-btn.selected-true {
            background: var(--accent-green);
            border-color: var(--accent-green);
            color: white;
        }

        .endpoint-btn.selected-false {
            background: var(--accent-red);
            border-color: var(--accent-red);
            color: white;
        }

        .endpoint-btn.selected-null {
            background: var(--bg-tertiary);
            border-color: var(--accent-purple);
            color: var(--accent-purple);
        }

        .visual-label-container {
            display: flex;
            gap: 8px;
        }

        .visual-input {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 14px;
        }

        .visual-input:focus {
            outline: none;
            border-color: var(--accent-blue);
        }

        .quick-labels {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }

        .quick-label {
            padding: 4px 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .quick-label:hover {
            background: var(--border-color);
        }

        .quick-label.active {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: white;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 12px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-secondary);
        }

        .status-dot.saved {
            background: var(--accent-green);
        }

        .status-dot.saving {
            background: var(--accent-yellow);
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .keyboard-hint {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 8px;
        }

        kbd {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 11px;
            font-family: inherit;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }

        .filter-bar {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .filter-btn {
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }

        .filter-btn:hover {
            color: var(--text-primary);
        }

        .filter-btn.active {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: white;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1>🎬 Segment Labeler</h1>
            
            <div class="progress-bar">
                <div class="progress-track">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <span class="progress-text" id="progressText">0 / 0 labeled</span>
            </div>

            <div class="nav-controls">
                <button class="nav-btn" id="prevBtn" onclick="navigateVideo(-1)">← Previous</button>
                <select class="video-select" id="videoSelect" onchange="loadVideoGroup(this.value)"></select>
                <button class="nav-btn" id="nextBtn" onclick="navigateVideo(1)">Next →</button>
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="filter-bar">
            <button class="filter-btn active" data-filter="all" onclick="setFilter('all')">All</button>
            <button class="filter-btn" data-filter="unlabeled" onclick="setFilter('unlabeled')">Unlabeled</button>
            <button class="filter-btn" data-filter="labeled" onclick="setFilter('labeled')">Labeled</button>
        </div>

        <div class="video-group-title" id="groupTitle">Loading...</div>
        
        <div class="segments-grid" id="segmentsGrid">
            <div class="empty-state">Loading segments...</div>
        </div>
    </div>

    <script>
        const VISUAL_LABELS = ['expressive', 'breath-in', 'mouth-open', 'nodding', 'looking-away', 'thinking'];
        
        let videoGroups = {};
        let entries = [];
        let currentGroupIndex = 0;
        let groupKeys = [];
        let currentFilter = 'all';

        async function init() {
            const response = await fetch('/api/data');
            const data = await response.json();
            entries = data.entries;
            videoGroups = data.video_groups;
            groupKeys = Object.keys(videoGroups).sort();
            
            // Populate video selector with status indicators
            const select = document.getElementById('videoSelect');
            select.innerHTML = groupKeys.map((key, idx) => {
                const group = videoGroups[key];
                const labeledCount = group.filter(i => entries[i].endpoint_bool !== null).length;
                const total = group.length;
                let status = '○'; // none labeled
                if (labeledCount === total) status = '✓'; // all labeled
                else if (labeledCount > 0) status = '◐'; // partial
                return `<option value="${idx}">${status} ${key} (${labeledCount}/${total})</option>`;
            }).join('');
            
            updateProgress();
            loadVideoGroup(0);
        }

        function updateProgress() {
            const labeled = entries.filter(e => e.endpoint_bool !== null).length;
            const total = entries.length;
            const pct = total > 0 ? (labeled / total * 100) : 0;
            
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressText').textContent = `${labeled} / ${total} labeled`;
            
            // Update select options with counts and status indicators
            const select = document.getElementById('videoSelect');
            groupKeys.forEach((key, idx) => {
                const group = videoGroups[key];
                const labeledCount = group.filter(i => entries[i].endpoint_bool !== null).length;
                const total = group.length;
                let status = '○'; // none labeled
                if (labeledCount === total) status = '✓'; // all labeled
                else if (labeledCount > 0) status = '◐'; // partial
                select.options[idx].textContent = `${status} ${key} (${labeledCount}/${total})`;
            });
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
            document.getElementById('groupTitle').textContent = groupKey;
            
            renderSegments();
        }

        function renderSegments() {
            const groupKey = groupKeys[currentGroupIndex];
            const segmentIndices = videoGroups[groupKey];
            
            let filteredIndices = segmentIndices;
            if (currentFilter === 'unlabeled') {
                filteredIndices = segmentIndices.filter(i => entries[i].endpoint_bool === null);
            } else if (currentFilter === 'labeled') {
                filteredIndices = segmentIndices.filter(i => entries[i].endpoint_bool !== null);
            }

            if (filteredIndices.length === 0) {
                document.getElementById('segmentsGrid').innerHTML = `
                    <div class="empty-state">No ${currentFilter} segments in this video</div>
                `;
                return;
            }

            document.getElementById('segmentsGrid').innerHTML = filteredIndices.map(idx => {
                const entry = entries[idx];
                const isLabeled = entry.endpoint_bool !== null;
                const hasVisual = entry.visual_label !== null && entry.visual_label !== '';
                const cardClass = isLabeled ? 'labeled' : (hasVisual ? 'partial' : '');
                
                const segmentName = entry.video_path.split('/').pop();
                
                return `
                    <div class="segment-card ${cardClass}" id="card-${idx}">
                        <div class="video-container">
                            <video controls preload="metadata">
                                <source src="/video/${entry.video_path}" type="video/mp4">
                            </video>
                        </div>
                        <div class="segment-info">
                            <div class="segment-name">${segmentName}</div>
                            
                            <div class="label-group">
                                <div class="label-title">End of Turn?</div>
                                <div class="endpoint-buttons">
                                    <button class="endpoint-btn ${entry.endpoint_bool === true ? 'selected-true' : ''}" 
                                            onclick="setEndpoint(${idx}, true)">✓ Yes</button>
                                    <button class="endpoint-btn ${entry.endpoint_bool === false ? 'selected-false' : ''}" 
                                            onclick="setEndpoint(${idx}, false)">✗ No</button>
                                    <button class="endpoint-btn ${entry.endpoint_bool === null ? 'selected-null' : ''}" 
                                            onclick="setEndpoint(${idx}, null)">Skip</button>
                                </div>
                            </div>
                            
                            <div class="label-group">
                                <div class="label-title">Visual Labels (optional, click to toggle)</div>
                                <div class="visual-label-container">
                                    <input type="text" class="visual-input" 
                                           value="${entry.visual_label || ''}"
                                           placeholder="e.g., expressive, breath-in"
                                           onchange="setVisualLabel(${idx}, this.value)"
                                           id="visual-${idx}">
                                </div>
                                <div class="quick-labels">
                                    ${(() => {
                                        const currentLabels = parseLabels(entry.visual_label);
                                        return VISUAL_LABELS.map(label => {
                                            const isActive = currentLabels.includes(label);
                                            return '<span class="quick-label ' + (isActive ? 'active' : '') + '" ' +
                                                  'data-label="' + label + '" ' +
                                                  'onclick="toggleVisualLabel(' + idx + ', \\'' + label + '\\')">' + label + '</span>';
                                        }).join('');
                                    })()}
                                    <span class="quick-label" onclick="setVisualLabel(${idx}, '')">clear all</span>
                                </div>
                            </div>
                            
                            <div class="status-indicator" id="status-${idx}">
                                <span class="status-dot ${isLabeled ? 'saved' : ''}"></span>
                                <span>${isLabeled ? 'Labeled' : 'Not labeled'}</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function setEndpoint(idx, value) {
            entries[idx].endpoint_bool = value;
            await saveEntry(idx);
            updateCardState(idx);
            updateProgress();
        }

        // Helper to parse comma-separated labels into array
        function parseLabels(labelStr) {
            if (!labelStr) return [];
            return labelStr.split(',').map(s => s.trim()).filter(s => s);
        }

        // Helper to format labels array to comma-separated string
        function formatLabels(labelsArray) {
            return labelsArray.length > 0 ? labelsArray.join(', ') : null;
        }

        // Toggle a visual label on/off
        async function toggleVisualLabel(idx, label) {
            const currentLabels = parseLabels(entries[idx].visual_label);
            const labelIndex = currentLabels.indexOf(label);
            
            if (labelIndex === -1) {
                currentLabels.push(label);
            } else {
                currentLabels.splice(labelIndex, 1);
            }
            
            entries[idx].visual_label = formatLabels(currentLabels);
            document.getElementById(`visual-${idx}`).value = entries[idx].visual_label || '';
            await saveEntry(idx);
            updateVisualLabelButtons(idx);
            updateCardState(idx);
        }

        // Set visual label directly (from text input)
        async function setVisualLabel(idx, value) {
            entries[idx].visual_label = value || null;
            await saveEntry(idx);
            updateVisualLabelButtons(idx);
            updateCardState(idx);
        }

        // Update quick label button states
        function updateVisualLabelButtons(idx) {
            const currentLabels = parseLabels(entries[idx].visual_label);
            document.querySelectorAll(`#card-${idx} .quick-label`).forEach(btn => {
                const label = btn.dataset.label;
                if (label) {
                    btn.classList.toggle('active', currentLabels.includes(label));
                }
            });
        }

        function updateCardState(idx) {
            const card = document.getElementById(`card-${idx}`);
            const entry = entries[idx];
            const isLabeled = entry.endpoint_bool !== null;
            const hasVisual = entry.visual_label !== null && entry.visual_label !== '';
            
            card.classList.remove('labeled', 'partial');
            if (isLabeled) card.classList.add('labeled');
            else if (hasVisual) card.classList.add('partial');
            
            // Update endpoint buttons
            card.querySelectorAll('.endpoint-btn').forEach(btn => {
                btn.classList.remove('selected-true', 'selected-false', 'selected-null');
            });
            if (entry.endpoint_bool === true) {
                card.querySelector('.endpoint-btn:nth-child(1)').classList.add('selected-true');
            } else if (entry.endpoint_bool === false) {
                card.querySelector('.endpoint-btn:nth-child(2)').classList.add('selected-false');
            } else {
                card.querySelector('.endpoint-btn:nth-child(3)').classList.add('selected-null');
            }
            
            // Update status
            const status = document.getElementById(`status-${idx}`);
            status.innerHTML = `
                <span class="status-dot saved"></span>
                <span>Saved</span>
            `;
        }

        async function saveEntry(idx) {
            const status = document.getElementById(`status-${idx}`);
            status.innerHTML = `
                <span class="status-dot saving"></span>
                <span>Saving...</span>
            `;
            
            try {
                await fetch('/api/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        idx: idx,
                        endpoint_bool: entries[idx].endpoint_bool,
                        visual_label: entries[idx].visual_label
                    })
                });
            } catch (e) {
                console.error('Save failed:', e);
                status.innerHTML = `
                    <span class="status-dot" style="background: var(--accent-red)"></span>
                    <span>Save failed</span>
                `;
            }
        }

        function navigateVideo(delta) {
            const newIndex = currentGroupIndex + delta;
            if (newIndex >= 0 && newIndex < groupKeys.length) {
                loadVideoGroup(newIndex);
            }
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            
            if (e.key === 'ArrowLeft') {
                navigateVideo(-1);
            } else if (e.key === 'ArrowRight') {
                navigateVideo(1);
            }
        });

        init();
    </script>
</body>
</html>
"""


def load_metadata():
    """Load metadata from JSONL file."""
    global ENTRIES
    ENTRIES = []

    if not METADATA_PATH.exists():
        return

    with open(METADATA_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ENTRIES.append(json.loads(line))
                except json.JSONDecodeError:
                    continue


def save_metadata():
    """Save metadata to JSONL file."""
    with open(METADATA_PATH, "w") as f:
        for entry in ENTRIES:
            f.write(json.dumps(entry) + "\n")


def group_by_video():
    """Group entries by original video (prefix before '_segment_')."""
    global VIDEO_GROUPS
    VIDEO_GROUPS = defaultdict(list)

    for idx, entry in enumerate(ENTRIES):
        video_path = entry.get("video_path", "")
        filename = os.path.basename(video_path)

        # Extract prefix before '_segment_'
        if "_segment_" in filename:
            prefix = filename.split("_segment_")[0]
        else:
            prefix = filename

        VIDEO_GROUPS[prefix].append(idx)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/data")
def get_data():
    load_metadata()
    group_by_video()
    return jsonify({"entries": ENTRIES, "video_groups": dict(VIDEO_GROUPS)})


@app.route("/api/update", methods=["POST"])
def update_entry():
    data = request.json
    idx = data["idx"]

    if 0 <= idx < len(ENTRIES):
        ENTRIES[idx]["endpoint_bool"] = data["endpoint_bool"]
        ENTRIES[idx]["visual_label"] = data["visual_label"]
        save_metadata()
        return jsonify({"success": True})

    return jsonify({"success": False, "error": "Invalid index"}), 400


@app.route("/video/<path:path>")
def serve_video(path):
    return send_from_directory(DATASET_DIR, path)


@app.route("/audio/<path:path>")
def serve_audio(path):
    return send_from_directory(DATASET_DIR / "audio", path)


def main():
    global DATASET_DIR, METADATA_PATH

    parser = argparse.ArgumentParser(
        description="Web-based labeling tool for video segments."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the dataset directory containing metadata.jsonl",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to run the server on (default: 5050)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    DATASET_DIR = Path(args.dataset_dir).resolve()
    METADATA_PATH = DATASET_DIR / "metadata.jsonl"

    if not METADATA_PATH.exists():
        print(f"❌ metadata.jsonl not found in {DATASET_DIR}")
        sys.exit(1)

    load_metadata()
    group_by_video()

    print(f"📂 Dataset: {DATASET_DIR}")
    print(f"📋 Entries: {len(ENTRIES)}")
    print(f"🎬 Video groups: {len(VIDEO_GROUPS)}")
    print(f"\n🌐 Open http://{args.host}:{args.port} in your browser\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
