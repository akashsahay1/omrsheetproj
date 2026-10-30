/**
 * OMR Processor - Main Application JavaScript
 */

// Application State
const AppState = {
    currentPage: 'dashboard',
    inputFolder: '',
    outputFile: '',
    fileCount: 0,
    answerKey: null,
    isProcessing: false,
    isPaused: false,
    results: [],
    progressInterval: null,
    examLevel: 'Prarambhik',
    examType: 'Chayan',
    sortColumn: 'rank',
    sortDirection: 'asc',
    engineReady: false,
    engineStatusInterval: null
};

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

// Wait for PyWebView API to be ready
window.addEventListener('pywebviewready', () => {
    console.log('PyWebView API ready');
    loadAnswerKeyStatus();
    loadSettings();
    // Start checking engine status
    startEngineStatusCheck();
});

/**
 * Initialize the application
 */
function initApp() {
    // Setup navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            navigateTo(page);
        });
    });

    // Initialize answer key grid
    initAnswerKeyGrid();

    // Show dashboard
    navigateTo('dashboard');
}

// ==================== Engine Status ====================

/**
 * Start checking ML engine initialization status
 */
function startEngineStatusCheck() {
    // Check immediately
    checkEngineStatus();

    // Then poll every 500ms until ready
    AppState.engineStatusInterval = setInterval(async () => {
        const ready = await checkEngineStatus();
        if (ready) {
            clearInterval(AppState.engineStatusInterval);
            AppState.engineStatusInterval = null;
        }
    }, 500);
}

/**
 * Check ML engine status and update UI
 * @returns {boolean} True if engine is ready
 */
async function checkEngineStatus() {
    try {
        const status = await API.getEngineStatus();

        if (status.initialized) {
            AppState.engineReady = true;
            updateEngineStatusUI(true, null);
            return true;
        } else if (status.error) {
            AppState.engineReady = false;
            updateEngineStatusUI(false, status.error);
            return true; // Stop polling on error
        } else if (status.initializing) {
            updateEngineStatusUI(false, 'loading');
            return false;
        }
    } catch (e) {
        console.log('Engine status check failed:', e);
    }
    return false;
}

/**
 * Update engine status in UI
 * @param {boolean} ready - Whether engine is ready
 * @param {string|null} error - Error message or 'loading' for loading state
 */
function updateEngineStatusUI(ready, error) {
    const statusBox = document.getElementById('engine-status');
    if (!statusBox) return;

    if (ready) {
        statusBox.className = 'status-box success';
        statusBox.innerHTML = 'ML Engine: Ready';
    } else if (error === 'loading') {
        statusBox.className = 'status-box warning';
        statusBox.innerHTML = 'ML Engine: Loading...';
    } else {
        statusBox.className = 'status-box error';
        statusBox.innerHTML = `ML Engine Error: ${error}`;
    }
}

/**
 * Navigate to a page
 * @param {string} page - Page ID
 */
function navigateTo(page) {
    // Update nav
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });

    // Update pages
    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === `page-${page}`);
    });

    AppState.currentPage = page;

    // Page-specific actions
    if (page === 'results') {
        loadResults();
    }
}

// ==================== File Selection ====================

/**
 * Select input folder
 */
async function selectInputFolder() {
    const result = await API.selectInputFolder();

    if (result.success) {
        AppState.inputFolder = result.path;
        AppState.fileCount = result.file_count;

        document.getElementById('input-folder').value = result.path;
        document.getElementById('file-count').textContent = `${result.file_count} images found`;

        showToast(`Selected folder with ${result.file_count} images`, 'success');
    } else {
        showToast(result.message || 'No folder selected', 'warning');
    }
}

/**
 * Select output file
 */
async function selectOutputFile() {
    const result = await API.selectOutputFile();

    if (result.success) {
        AppState.outputFile = result.path;
        document.getElementById('output-file').value = result.path;
    }
}

// ==================== Answer Key ====================

/**
 * Initialize answer key grid
 */
function initAnswerKeyGrid() {
    const grid = document.getElementById('answer-key-grid');
    grid.innerHTML = '';

    for (let q = 1; q <= 60; q++) {
        const item = document.createElement('div');
        item.className = 'answer-key-item';
        item.innerHTML = `
            <label>Q${q}</label>
            <select id="ak-${q}" onchange="updateAnswerKeyCount()">
                <option value="">-</option>
                <option value="A">A</option>
                <option value="B">B</option>
                <option value="C">C</option>
                <option value="D">D</option>
            </select>
        `;
        grid.appendChild(item);
    }
}

/**
 * Load answer key from file
 */
async function loadAnswerKeyFile() {
    const result = await API.selectAnswerKeyFile();

    if (result.success) {
        AppState.answerKey = result.answers;

        // Update grid
        for (let q = 1; q <= 60; q++) {
            const select = document.getElementById(`ak-${q}`);
            if (select) {
                select.value = result.answers[q] || '';
            }
        }

        updateAnswerKeyCount();
        updateAnswerKeyStatus();
        showToast('Answer key loaded successfully', 'success');
    } else {
        showToast(result.message || 'Failed to load answer key', 'error');
    }
}

/**
 * Save answer key from grid
 */
async function saveAnswerKey() {
    const answers = {};

    for (let q = 1; q <= 60; q++) {
        const select = document.getElementById(`ak-${q}`);
        if (select && select.value) {
            answers[q] = select.value;
        }
    }

    const result = await API.setManualAnswerKey(answers);

    if (result.success) {
        AppState.answerKey = answers;
        updateAnswerKeyStatus();
        showToast('Answer key saved', 'success');
    } else {
        showToast(result.message || 'Failed to save answer key', 'error');
    }
}

/**
 * Update answer key count display
 */
function updateAnswerKeyCount() {
    let count = 0;
    for (let q = 1; q <= 60; q++) {
        const select = document.getElementById(`ak-${q}`);
        if (select && select.value) {
            count++;
        }
    }
    document.getElementById('ak-filled').textContent = count;
}

/**
 * Load answer key status
 */
async function loadAnswerKeyStatus() {
    // First, try to load saved answer key from previous session
    try {
        const savedResult = await API.loadSavedAnswerKey();

        if (savedResult.success) {
            console.log('Loaded saved answer key:', savedResult.name || 'Unnamed');
            AppState.answerKey = savedResult.answers;

            // Update grid with saved answers
            for (let q = 1; q <= 60; q++) {
                const select = document.getElementById(`ak-${q}`);
                if (select && savedResult.answers[q]) {
                    select.value = savedResult.answers[q];
                }
            }

            updateAnswerKeyCount();
            updateAnswerKeyStatus();
            return;
        }
    } catch (e) {
        console.log('No saved answer key found:', e);
    }

    // Fall back to checking current answer key
    const result = await API.getAnswerKey();

    if (result.success) {
        AppState.answerKey = result.answers;

        // Update grid
        for (let q = 1; q <= 60; q++) {
            const select = document.getElementById(`ak-${q}`);
            if (select && result.answers[q]) {
                select.value = result.answers[q];
            }
        }

        updateAnswerKeyCount();
    }

    updateAnswerKeyStatus();
}

/**
 * Update answer key status display
 */
function updateAnswerKeyStatus() {
    const statusBox = document.getElementById('answer-key-status');

    if (AppState.answerKey && Object.keys(AppState.answerKey).length > 0) {
        const count = Object.values(AppState.answerKey).filter(v => v).length;
        statusBox.className = 'status-box ' + (count === 60 ? 'success' : 'warning');
        statusBox.textContent = `Answer key loaded: ${count}/60 answers`;
    } else {
        statusBox.className = 'status-box warning';
        statusBox.textContent = 'No answer key loaded';
    }
}

// ==================== Processing ====================

/**
 * Start or toggle pause for batch processing
 */
async function startProcessing() {
    // If processing and not paused, pause it
    if (AppState.isProcessing && !AppState.isPaused) {
        const result = await API.pauseProcessing();
        if (result.success) {
            AppState.isPaused = true;
            document.getElementById('btn-start').textContent = 'Resume';
            showToast('Processing paused', 'warning');
        }
        return;
    }

    // If paused, resume it
    if (AppState.isProcessing && AppState.isPaused) {
        const result = await API.resumeProcessing();
        if (result.success) {
            AppState.isPaused = false;
            document.getElementById('btn-start').textContent = 'Pause';
            showToast('Processing resumed', 'success');
        }
        return;
    }

    // Validate for new processing
    if (!AppState.inputFolder) {
        showToast('Please select an input folder', 'warning');
        return;
    }

    const withScoring = document.getElementById('with-scoring').checked;

    if (withScoring && (!AppState.answerKey || Object.keys(AppState.answerKey).length === 0)) {
        showToast('Please load an answer key for scoring', 'warning');
        return;
    }

    // Get exam type selections
    AppState.examLevel = document.getElementById('exam-level').value;
    AppState.examType = document.getElementById('exam-type').value;

    // Start processing
    const result = await API.startProcessing(
        AppState.inputFolder,
        AppState.outputFile || null,
        withScoring,
        AppState.examLevel,
        AppState.examType
    );

    if (result.success) {
        AppState.isProcessing = true;
        AppState.isPaused = false;

        // Update UI
        document.getElementById('btn-start').textContent = 'Pause';
        document.getElementById('btn-cancel').disabled = false;
        document.getElementById('progress-section').classList.remove('hidden');

        // Start progress polling
        startProgressPolling();

        showToast('Processing started', 'success');
    } else {
        showToast(result.message || 'Failed to start processing', 'error');
    }
}

/**
 * Cancel processing and reset everything
 */
async function cancelProcessing() {
    const result = await API.cancelProcessing();

    if (result.success) {
        // Stop polling
        stopProgressPolling();

        // Reset state
        AppState.isProcessing = false;
        AppState.isPaused = false;
        AppState.results = [];

        // Reset UI
        document.getElementById('btn-start').textContent = 'Start Scan';
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-cancel').disabled = true;

        // Reset progress bar
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        document.getElementById('progress-status').textContent = 'Status: Ready';
        document.getElementById('progress-file').textContent = 'Current: -';
        document.getElementById('progress-count').textContent = 'Processed: 0 / 0';
        document.getElementById('progress-time').textContent = 'Elapsed: 0s | Remaining: -';
        document.getElementById('progress-section').classList.add('hidden');

        // Reset results display to initial state
        document.getElementById('results-stats').classList.add('hidden');
        document.getElementById('results-empty').classList.remove('hidden');
        document.getElementById('results-table-container').classList.add('hidden');
        document.getElementById('results-body').innerHTML = '';

        // Reset stats values
        document.getElementById('stat-total').textContent = '0';
        document.getElementById('stat-avg').textContent = '0';
        document.getElementById('stat-high').textContent = '0';
        document.getElementById('stat-low').textContent = '0';

        showToast('Processing cancelled and reset', 'warning');
    }
}

/**
 * Start polling for progress updates
 */
function startProgressPolling() {
    if (AppState.progressInterval) {
        clearInterval(AppState.progressInterval);
    }

    AppState.progressInterval = setInterval(async () => {
        const progress = await API.getProgress();
        updateProgressUI(progress);

        // Check if completed or cancelled
        if (progress.status === 'completed' || progress.status === 'cancelled' || progress.status === 'error') {
            stopProgressPolling();
            AppState.isProcessing = false;
            AppState.isPaused = false;

            document.getElementById('btn-start').textContent = 'Start Scan';
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-cancel').disabled = true;

            if (progress.status === 'completed') {
                showToast('Processing completed!', 'success');
                loadResults();
            } else if (progress.status === 'error') {
                showToast(`Error: ${progress.message}`, 'error');
            }
        }

        // Update button text based on paused state
        if (progress.status === 'paused') {
            AppState.isPaused = true;
            document.getElementById('btn-start').textContent = 'Resume';
        } else if (progress.status === 'running') {
            AppState.isPaused = false;
            document.getElementById('btn-start').textContent = 'Pause';
        }
    }, 500);
}

/**
 * Stop progress polling
 */
function stopProgressPolling() {
    if (AppState.progressInterval) {
        clearInterval(AppState.progressInterval);
        AppState.progressInterval = null;
    }
}

/**
 * Update progress UI
 * @param {Object} progress - Progress data
 */
function updateProgressUI(progress) {
    document.getElementById('progress-bar').style.width = `${progress.percentage}%`;
    document.getElementById('progress-text').textContent = `${Math.round(progress.percentage)}%`;
    document.getElementById('progress-status').textContent = `Status: ${progress.status}`;
    document.getElementById('progress-file').textContent = `Current: ${progress.current_file || '-'}`;
    document.getElementById('progress-count').textContent = `Processed: ${progress.current} / ${progress.total}`;

    const elapsed = formatDuration(progress.elapsed_seconds);
    const remaining = progress.estimated_remaining > 0 ? formatDuration(progress.estimated_remaining) : '-';
    document.getElementById('progress-time').textContent = `Elapsed: ${elapsed} | Remaining: ${remaining}`;
}

/**
 * Format seconds as "1h 23m 45s", "4m 05s" or "12s"
 * @param {number} totalSeconds
 * @returns {string}
 */
function formatDuration(totalSeconds) {
    const s = Math.max(0, Math.round(totalSeconds || 0));
    const hours = Math.floor(s / 3600);
    const minutes = Math.floor((s % 3600) / 60);
    const seconds = s % 60;
    const pad = n => String(n).padStart(2, '0');
    if (hours > 0) return `${hours}h ${pad(minutes)}m ${pad(seconds)}s`;
    if (minutes > 0) return `${minutes}m ${pad(seconds)}s`;
    return `${seconds}s`;
}

// ==================== Results ====================

/**
 * Load and display results
 */
async function loadResults() {
    const result = await API.getResults();

    if (result.success && result.results.length > 0) {
        AppState.results = result.results;

        // Update stats
        document.getElementById('stat-total').textContent = result.total_count;
        document.getElementById('stat-avg').textContent = result.statistics.average_score;
        document.getElementById('stat-high').textContent = result.statistics.highest_score;
        document.getElementById('stat-low').textContent = result.statistics.lowest_score;

        // Show export path notification if available
        if (result.output_path) {
            showToast(`Results exported to: ${result.output_path}`, 'success');
        }

        // Show stats bar
        document.getElementById('results-stats').classList.remove('hidden');
        document.getElementById('results-empty').classList.add('hidden');
        document.getElementById('results-table-container').classList.remove('hidden');

        // Populate table
        const tbody = document.getElementById('results-body');
        tbody.innerHTML = '';

        // Sort results based on current sort settings
        const sortedResults = sortResultsArray(result.results);

        sortedResults.forEach((r, index) => {
            const row = document.createElement('tr');
            row.className = 'result-row' + (r.multi_marked > 0 ? ' multi-marked' : '');
            row.onclick = () => showResultDetail(r._originalIndex);

            // Confidence display with color coding
            const confidence = r.confidence !== undefined ? r.confidence : 100;
            const confClass = confidence >= 90 ? 'conf-high' : (confidence >= 70 ? 'conf-medium' : 'conf-low');

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${r.rank}</td>
                <td class="filename-cell" title="${r.filename || ''}">${r.filename || ''}</td>
                <td>${r.name}</td>
                <td>${r.roll_number}</td>
                <td>${r.class}</td>
                <td>${r.section}</td>
                <td>${r.school_code}</td>
                <td>${r.correct}</td>
                <td>${r.wrong}</td>
                <td>${r.unanswered}</td>
                <td>${r.total_score}</td>
                <td>${r.percentage}%</td>
                <td class="${confClass}">${confidence.toFixed(1)}%</td>
            `;

            tbody.appendChild(row);
        });

        // Update sort icons
        updateSortIcons();
    } else {
        // Show empty state
        document.getElementById('results-stats').classList.add('hidden');
        document.getElementById('results-empty').classList.remove('hidden');
        document.getElementById('results-table-container').classList.add('hidden');
    }
}

/**
 * Show result detail in modal
 * @param {number} index - Result index
 */
async function showResultDetail(index) {
    const result = await API.getResultDetail(index);

    if (result.success) {
        const candidate = result.candidate;
        const evaluation = result.evaluation;

        const modalBody = document.getElementById('modal-body');
        modalBody.innerHTML = `
            <div class="candidate-detail">
                <div class="detail-section">
                    <h4>Candidate Information</h4>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <span class="detail-label">Name</span>
                            <span class="detail-value">${candidate.name}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Roll Number</span>
                            <span class="detail-value">${candidate.roll_number}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Class</span>
                            <span class="detail-value">${candidate.class}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Section</span>
                            <span class="detail-value">${candidate.section}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">School Code</span>
                            <span class="detail-value">${candidate.school_code}</span>
                        </div>
                    </div>
                </div>

                <div class="detail-section">
                    <h4>Score Summary</h4>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <span class="detail-label">Total Score</span>
                            <span class="detail-value">${evaluation.total_score}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Percentage</span>
                            <span class="detail-value">${evaluation.percentage}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Rank</span>
                            <span class="detail-value">${evaluation.rank}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Correct</span>
                            <span class="detail-value">${evaluation.correct_count}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Wrong</span>
                            <span class="detail-value">${evaluation.wrong_count}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Unanswered</span>
                            <span class="detail-value">${evaluation.unanswered_count}</span>
                        </div>
                    </div>
                </div>

                <div class="detail-section">
                    <h4>Answers</h4>
                    <div class="answers-grid">
                        ${generateAnswersGrid(candidate.answers, evaluation.question_results)}
                    </div>
                </div>
            </div>
        `;

        openModal();
    }
}

/**
 * Generate answers grid HTML
 */
function generateAnswersGrid(answers, questionResults) {
    let html = '';

    for (let q = 1; q <= 60; q++) {
        const answer = answers[q] || '-';
        const result = questionResults[q] || 'unanswered';
        html += `<div class="answer-cell ${result}">${q}: ${answer}</div>`;
    }

    return html;
}

/**
 * Export results to Excel
 */
async function exportResults() {
    const result = await API.selectOutputFile();

    if (result.success) {
        // Get current exam selections
        const examLevel = document.getElementById('exam-level')?.value || AppState.examLevel;
        const examType = document.getElementById('exam-type')?.value || AppState.examType;

        const exportResult = await API.exportResults(result.path, examLevel, examType);

        if (exportResult.success) {
            showToast('Results exported successfully', 'success');
        } else {
            showToast(exportResult.message || 'Export failed', 'error');
        }
    }
}

// ==================== Sorting ====================

/**
 * Sort results by column
 * @param {string} column - Column to sort by
 */
function sortResults(column) {
    // Toggle direction if same column, otherwise default to ascending
    if (AppState.sortColumn === column) {
        AppState.sortDirection = AppState.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        AppState.sortColumn = column;
        AppState.sortDirection = 'asc';
    }

    // Re-render results table
    renderResultsTable();
}

/**
 * Sort results array based on current sort settings
 * @param {Array} results - Results array to sort
 * @returns {Array} Sorted array with original indices preserved
 */
function sortResultsArray(results) {
    // Add original index for detail view
    const indexed = results.map((r, i) => ({ ...r, _originalIndex: i }));

    return indexed.sort((a, b) => {
        let valA, valB;

        switch (AppState.sortColumn) {
            case 'rank':
                valA = parseInt(a.rank) || 0;
                valB = parseInt(b.rank) || 0;
                break;
            case 'name':
                valA = (a.name || '').toLowerCase();
                valB = (b.name || '').toLowerCase();
                break;
            case 'total_score':
                valA = parseFloat(a.total_score) || 0;
                valB = parseFloat(b.total_score) || 0;
                break;
            case 'percentage':
                valA = parseFloat(a.percentage) || 0;
                valB = parseFloat(b.percentage) || 0;
                break;
            case 'confidence':
                valA = parseFloat(a.confidence !== undefined ? a.confidence : 100);
                valB = parseFloat(b.confidence !== undefined ? b.confidence : 100);
                break;
            default:
                valA = a[AppState.sortColumn];
                valB = b[AppState.sortColumn];
        }

        // Compare values
        let comparison = 0;
        if (typeof valA === 'string') {
            comparison = valA.localeCompare(valB);
        } else {
            comparison = valA - valB;
        }

        return AppState.sortDirection === 'asc' ? comparison : -comparison;
    });
}

/**
 * Render results table with current sort
 */
function renderResultsTable() {
    if (!AppState.results || AppState.results.length === 0) return;

    const tbody = document.getElementById('results-body');
    tbody.innerHTML = '';

    const sortedResults = sortResultsArray(AppState.results);

    sortedResults.forEach((r, index) => {
        const row = document.createElement('tr');
        row.className = 'result-row' + (r.multi_marked > 0 ? ' multi-marked' : '');
        row.onclick = () => showResultDetail(r._originalIndex);

        const confidence = r.confidence !== undefined ? r.confidence : 100;
        const confClass = confidence >= 90 ? 'conf-high' : (confidence >= 70 ? 'conf-medium' : 'conf-low');

        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${r.rank}</td>
            <td class="filename-cell" title="${r.filename || ''}">${r.filename || ''}</td>
            <td>${r.name}</td>
            <td>${r.roll_number}</td>
            <td>${r.class}</td>
            <td>${r.section}</td>
            <td>${r.school_code}</td>
            <td>${r.correct}</td>
            <td>${r.wrong}</td>
            <td>${r.unanswered}</td>
            <td>${r.total_score}</td>
            <td>${r.percentage}%</td>
            <td class="${confClass}">${confidence.toFixed(1)}%</td>
        `;

        tbody.appendChild(row);
    });

    updateSortIcons();
}

/**
 * Update sort icons to show current sort state
 */
function updateSortIcons() {
    // Reset all sort icons
    document.querySelectorAll('.sortable .sort-icon').forEach(icon => {
        icon.innerHTML = '&#8597;'; // Up-down arrow
        icon.parentElement.classList.remove('sort-asc', 'sort-desc');
    });

    // Update current sort column icon
    const currentHeader = document.querySelector(`.sortable[data-sort="${AppState.sortColumn}"]`);
    if (currentHeader) {
        const icon = currentHeader.querySelector('.sort-icon');
        if (icon) {
            icon.innerHTML = AppState.sortDirection === 'asc' ? '&#8593;' : '&#8595;'; // Up or down arrow
        }
        currentHeader.classList.add(AppState.sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
    }
}

// ==================== Settings ====================

/**
 * Load settings
 */
async function loadSettings() {
    const result = await API.getSettings();

    if (result.success) {
        const settings = result.settings;

        document.getElementById('setting-correct').value = settings.scoring.correct_marks;
        document.getElementById('setting-wrong').value = settings.scoring.wrong_marks;
        document.getElementById('setting-unanswered').value = settings.scoring.unanswered_marks;
    }
}

/**
 * Save settings
 */
async function saveSettings() {
    const settings = {
        scoring: {
            correct_marks: parseFloat(document.getElementById('setting-correct').value),
            wrong_marks: parseFloat(document.getElementById('setting-wrong').value),
            unanswered_marks: parseFloat(document.getElementById('setting-unanswered').value)
        }
    };

    const result = await API.saveSettings(settings);

    if (result.success) {
        showToast('Settings saved', 'success');
    } else {
        showToast(result.message || 'Failed to save settings', 'error');
    }
}

// ==================== Modal ====================

function openModal() {
    document.getElementById('modal-overlay').classList.remove('hidden');
}

function closeModal() {
    document.getElementById('modal-overlay').classList.add('hidden');
}

// Close modal on overlay click
document.getElementById('modal-overlay')?.addEventListener('click', (e) => {
    if (e.target.id === 'modal-overlay') {
        closeModal();
    }
});

// ==================== Toast Notifications ====================

/**
 * Show toast notification
 * @param {string} message - Message to display
 * @param {string} type - Toast type (success, error, warning)
 */
function showToast(message, type = 'info') {
    // Create container if doesn't exist
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    // Create toast
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Remove after delay
    setTimeout(() => {
        toast.remove();
    }, 4000);
}

// Make functions globally available
window.navigateTo = navigateTo;
window.selectInputFolder = selectInputFolder;
window.selectOutputFile = selectOutputFile;
window.loadAnswerKeyFile = loadAnswerKeyFile;
window.saveAnswerKey = saveAnswerKey;
window.startProcessing = startProcessing;
window.cancelProcessing = cancelProcessing;
window.exportResults = exportResults;
window.saveSettings = saveSettings;
window.showResultDetail = showResultDetail;
window.openModal = openModal;
window.closeModal = closeModal;
window.updateAnswerKeyCount = updateAnswerKeyCount;
window.sortResults = sortResults;
