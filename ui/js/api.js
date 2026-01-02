/**
 * API wrapper for Python backend communication
 */

const API = {
    /**
     * Call Python API method
     * @param {string} method - Method name
     * @param {any[]} args - Arguments
     * @returns {Promise<any>}
     */
    async call(method, ...args) {
        try {
            if (window.pywebview && window.pywebview.api) {
                const result = await window.pywebview.api[method](...args);
                return result;
            } else {
                console.error('PyWebView API not available');
                return { success: false, message: 'API not available' };
            }
        } catch (error) {
            console.error(`API call failed: ${method}`, error);
            return { success: false, message: error.message };
        }
    },

    // Application Info
    async getAppInfo() {
        return this.call('get_app_info');
    },

    // Engine Initialization
    async startEngineInit() {
        return this.call('start_engine_init');
    },

    async getEngineStatus() {
        return this.call('get_engine_status');
    },

    // File Dialogs
    async selectInputFolder() {
        return this.call('select_input_folder');
    },

    async selectOutputFile() {
        return this.call('select_output_file');
    },

    async selectAnswerKeyFile() {
        return this.call('select_answer_key_file');
    },

    // Answer Key
    async loadAnswerKey(filePath) {
        return this.call('load_answer_key', filePath);
    },

    async setManualAnswerKey(answers) {
        return this.call('set_manual_answer_key', answers);
    },

    async getAnswerKey() {
        return this.call('get_answer_key');
    },

    async loadSavedAnswerKey() {
        return this.call('load_saved_answer_key');
    },

    async clearSavedAnswerKey() {
        return this.call('clear_saved_answer_key');
    },

    // Processing
    async startProcessing(inputFolder, outputPath, withScoring, examLevel, examType) {
        return this.call('start_processing', inputFolder, outputPath, withScoring, examLevel, examType);
    },

    async cancelProcessing() {
        return this.call('cancel_processing');
    },

    async pauseProcessing() {
        return this.call('pause_processing');
    },

    async resumeProcessing() {
        return this.call('resume_processing');
    },

    async getProgress() {
        return this.call('get_progress');
    },

    async isProcessing() {
        return this.call('is_processing');
    },

    // Results
    async getResults() {
        return this.call('get_results');
    },

    async getResultDetail(index) {
        return this.call('get_result_detail', index);
    },

    async exportResults(outputPath, examLevel, examType) {
        return this.call('export_results', outputPath, examLevel, examType);
    },

    // Settings
    async getSettings() {
        return this.call('get_settings');
    },

    async saveSettings(settings) {
        return this.call('save_settings', settings);
    },

    // Single Image
    async processSingleImage(imagePath) {
        return this.call('process_single_image', imagePath);
    },

    async selectSingleImage() {
        return this.call('select_single_image');
    }
};

// Export for use
window.API = API;
