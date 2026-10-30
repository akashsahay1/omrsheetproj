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

    // Answer Keys (saved by exam name)
    async listAnswerKeys() {
        return this.call('list_answer_keys');
    },

    async createAnswerKey(name) {
        return this.call('create_answer_key', name);
    },

    async getSavedAnswerKey(keyId) {
        return this.call('get_saved_answer_key', keyId);
    },

    async updateAnswerKey(keyId, name, answers) {
        return this.call('update_answer_key', keyId, name, answers);
    },

    async deleteAnswerKey(keyId) {
        return this.call('delete_answer_key', keyId);
    },

    async selectAnswerKey(keyId) {
        return this.call('select_answer_key', keyId);
    },

    async importAnswersFromFile() {
        return this.call('import_answers_from_file');
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

    // School Codes
    async getSchoolCodes() {
        return this.call('get_school_codes');
    },

    async saveSchoolCode(code, name, originalCode) {
        return this.call('save_school_code', code, name, originalCode);
    },

    async deleteSchoolCode(code) {
        return this.call('delete_school_code', code);
    },

    async importSchoolCodesFile() {
        return this.call('import_school_codes_file');
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
