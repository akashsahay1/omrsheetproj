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
    results: [],
    progressInterval: null,
    examLevel: 'Prarambhik',
    examType: 'Chayan'
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
 * Start batch processing
 */
async function startProcessing() {
    // Validate
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

        // Update UI
        document.getElementById('btn-start').disabled = true;
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
 * Cancel processing
 */
async function cancelProcessing() {
    const result = await API.cancelProcessing();

    if (result.success) {
        showToast('Cancellation requested', 'warning');
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

            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-cancel').disabled = true;

            if (progress.status === 'completed') {
                showToast('Processing completed!', 'success');
                loadResults();
            } else if (progress.status === 'error') {
                showToast(`Error: ${progress.message}`, 'error');
            }
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

    const elapsed = Math.round(progress.elapsed_seconds);
    const remaining = progress.estimated_remaining > 0 ? Math.round(progress.estimated_remaining) : '-';
    document.getElementById('progress-time').textContent = `Elapsed: ${elapsed}s | Remaining: ${remaining}s`;
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

        // Show stats bar
        document.getElementById('results-stats').classList.remove('hidden');
        document.getElementById('results-empty').classList.add('hidden');
        document.getElementById('results-table-container').classList.remove('hidden');

        // Populate table
        const tbody = document.getElementById('results-body');
        tbody.innerHTML = '';

        result.results.forEach((r, index) => {
            const row = document.createElement('tr');
            row.className = 'result-row' + (r.multi_marked > 0 ? ' multi-marked' : '');
            row.onclick = () => showResultDetail(index);

            row.innerHTML = `
                <td>${r.rank}</td>
                <td class="filename-cell" title="${r.filename || ''}">${r.filename || ''}</td>
                <td>${r.name}</td>
                <td>${r.roll_number}</td>
                <td>${r.class}</td>
                <td>${r.section}</td>
                <td>${r.school_code}</td>
                <td>${r.correct}</td>
                <td>${r.wrong}</td>
                <td>${r.total_score}</td>
                <td>${r.percentage}%</td>
            `;

            tbody.appendChild(row);
        });
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
